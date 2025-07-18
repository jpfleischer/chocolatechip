#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

MAX_BATCH_SIZE = 1

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model back to Darknet weights (reverse conversion).")
    parser.add_argument('-c', '--config', required=True,
                        help='Path to Darknet .cfg file')
    parser.add_argument('-i', '--onnx', required=True,
                        help='Path to ONNX model file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output path for Darknet .weights file')
    return parser.parse_args()

def rreplace(s, old, new, occurrence=1):
    return new.join(s.rsplit(old, occurrence))

def is_pan_arch(cfg_file_path):
    with open(cfg_file_path, 'r') as f:
        cfg_lines = [l.strip() for l in f.readlines()]
    yolos_or_upsamples = [l for l in cfg_lines if l in ['[yolo]', '[upsample]']]
    yolo_count = len([l for l in yolos_or_upsamples if l == '[yolo]'])
    upsample_count = len(yolos_or_upsamples) - yolo_count
    assert 2 <= yolo_count <= 4
    assert upsample_count == yolo_count - 1 or upsample_count == 0
    return yolos_or_upsamples[0] == '[upsample]'

def get_output_convs(layer_configs):
    """
    Find the output convolutional layer names based on the Darknet cfg.
    (These names were used during conversion.)
    """
    output_convs = []
    previous_layer = None
    for current_layer in layer_configs.keys():
        if previous_layer is not None and current_layer.endswith('yolo'):
            assert previous_layer.endswith('convolutional')
            activation = layer_configs[previous_layer]['activation']
            if activation == 'linear':
                output_convs.append(previous_layer)
            elif activation == 'logistic':
                output_convs.append(previous_layer + '_lgx')
            else:
                raise TypeError('unexpected activation: %s' % activation)
        previous_layer = current_layer
    return output_convs

def get_category_num(cfg_file_path):
    with open(cfg_file_path, 'r') as f:
        cfg_lines = [l.strip() for l in f.readlines()]
    classes_lines = [l for l in cfg_lines if l.startswith('classes=')]
    assert len(set(classes_lines)) == 1
    return int(classes_lines[-1].split('=')[-1].strip())

def get_h_and_w(layer_configs):
    net_config = layer_configs['000_net']
    return net_config['height'], net_config['width']

# === DARKNET PARSER AND GRAPH BUILDER CLASSES ===

class DarkNetParser(object):
    """Parses the Darknet .cfg file and returns an OrderedDict of layer configurations."""
    def __init__(self, supported_layers=None):
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers if supported_layers else \
            ['net', 'convolutional', 'maxpool', 'shortcut',
             'route', 'upsample', 'yolo']
        self.layer_counter = 0

    def parse_cfg_file(self, cfg_file_path):
        with open(cfg_file_path, 'r') as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs

    def _next_layer(self, remainder):
        remainder = remainder.split('[', 1)
        while len(remainder[0]) > 0 and remainder[0][-1] == '#':
            remainder = remainder[1].split('[', 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None
        remainder = remainder.split(']', 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            raise ValueError('no closing bracket!')
        if layer_type not in self.supported_layers:
            raise ValueError('%s layer not supported!' % layer_type)

        out = remainder.split('\n[', 1)
        if len(out) == 2:
            layer_param_block, remainder = out[0], '[' + out[1]
        else:
            layer_param_block, remainder = out[0], ''
        layer_param_lines = layer_param_block.split('\n')
        layer_param_lines = [l.lstrip() for l in layer_param_lines if l.lstrip()]
        # Do not parse yolo layer parameters.
        if layer_type == 'yolo':
            layer_param_lines = []
        skip_params = ['steps', 'scales'] if layer_type == 'net' else []
        layer_name = str(self.layer_counter).zfill(3) + '_' + layer_type
        layer_dict = dict(type=layer_type)
        for param_line in layer_param_lines:
            param_line = param_line.split('#')[0]
            if not param_line:
                continue
            assert '[' not in param_line
            param_type, param_value = self._parse_params(param_line, skip_params)
            if param_type is not None:
                layer_dict[param_type] = param_value
        self.layer_counter += 1
        return layer_dict, layer_name, remainder

    def _parse_params(self, param_line, skip_params=None):
        param_line = param_line.replace(' ', '')
        param_type, param_value_raw = param_line.split('=')
        assert param_value_raw
        param_value = None
        if skip_params and param_type in skip_params:
            param_type = None
        elif param_type == 'layers':
            layer_indexes = list()
            for index in param_value_raw.split(','):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = param_value_raw[0] == '-' and param_value_raw[1:].isdigit()
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)
        return param_type, param_value

class MajorNodeSpecs(object):
    """Holds the name and output channels for a node."""
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels
        self.created_onnx_node = False
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True

class ConvParams(object):
    """Stores convolutional layer parameters for weight ordering."""
    def __init__(self, node_name, batch_normalize, conv_weight_dims):
        self.node_name = node_name
        self.batch_normalize = batch_normalize
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims

    def generate_param_name(self, param_category, suffix):
        """Generates the parameter name as used in the ONNX conversion."""
        assert suffix
        assert param_category in ['bn', 'conv']
        assert(suffix in ['scale', 'mean', 'var', 'weights', 'bias'])
        if param_category == 'bn':
            assert self.batch_normalize
            assert suffix in ['scale', 'bias', 'mean', 'var']
        elif param_category == 'conv':
            assert suffix in ['weights', 'bias']
            if suffix == 'bias':
                assert not self.batch_normalize
        param_name = self.node_name + '_' + param_category + '_' + suffix
        return param_name

class UpsampleParams(object):
    """Stores the upsample scale (unused in Darknet weights)."""
    def __init__(self, node_name, value):
        self.node_name = node_name
        self.value = value

    def generate_param_name(self):
        param_name = self.node_name + '_' + 'scale'
        return param_name

class GraphBuilderONNX(object):
    """
    A dummy ONNX graph builder used here only to reconstruct the order
    and naming of parameters as produced during conversion.
    """
    def __init__(self, model_name, output_tensors, batch_size):
        self.model_name = model_name
        self.output_tensors = output_tensors
        self._nodes = list()
        self.graph_def = None
        self.input_tensor = None
        self.epsilon_bn = 1e-5
        self.momentum_bn = 0.99
        self.alpha_lrelu = 0.1
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.batch_size = batch_size
        self.route_spec = 0

    def build_dummy_graph(self, layer_configs):
        # Iterate over layers to create ONNX nodes (only for ordering and param_dict)
        for layer_name in layer_configs.keys():
            layer_dict = layer_configs[layer_name]
            major_node_specs = self._make_onnx_node(layer_name, layer_dict)
            if major_node_specs.name is not None:
                self.major_node_specs.append(major_node_specs)
        # Remove dummy nodes
        self.major_node_specs = [node for node in self.major_node_specs if 'dummy' not in node.name]

    def _make_onnx_node(self, layer_name, layer_dict):
        layer_type = layer_dict['type']
        if self.input_tensor is None:
            if layer_type == 'net':
                major_node_output_name, major_node_output_channels = self._make_input_tensor(layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name, major_node_output_channels)
            else:
                raise ValueError('The first node has to be of type "net".')
        else:
            node_creators = {
                'convolutional': self._make_conv_node,
                'maxpool': self._make_maxpool_node,
                'shortcut': self._make_shortcut_node,
                'route': self._make_route_node,
                'upsample': self._make_upsample_node,
                'yolo': self._make_yolo_node
            }
            if layer_type in node_creators.keys():
                major_node_output_name, major_node_output_channels = node_creators[layer_type](layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name, major_node_output_channels)
            else:
                raise TypeError('layer of type %s not supported' % layer_type)
        return major_node_specs

    def _make_input_tensor(self, layer_name, layer_dict):
        channels = layer_dict['channels']
        height = layer_dict['height']
        width = layer_dict['width']
        input_tensor = helper.make_tensor_value_info(str(layer_name), TensorProto.FLOAT,
                                                     [self.batch_size, channels, height, width])
        self.input_tensor = input_tensor
        return layer_name, channels

    def _get_previous_node_specs(self, target_index=0):
        if target_index == 0:
            if self.route_spec != 0:
                previous_node = self.major_node_specs[self.route_spec]
                self.route_spec = 0
            else:
                previous_node = self.major_node_specs[-1]
        else:
            previous_node = self.major_node_specs[target_index]
        return previous_node

    def _make_conv_node(self, layer_name, layer_dict):
        previous_node_specs = self._get_previous_node_specs()
        previous_channels = previous_node_specs.channels
        kernel_size = layer_dict['size']
        stride = layer_dict['stride']
        filters = layer_dict['filters']
        batch_normalize = ('batch_normalize' in layer_dict and layer_dict['batch_normalize'] == 1)
        kernel_shape = [kernel_size, kernel_size]
        weights_shape = [filters, previous_channels] + kernel_shape
        conv_params = ConvParams(layer_name, batch_normalize, weights_shape)
        # Create a dummy conv node (for ordering)
        conv_node = helper.make_node(
            'Conv',
            inputs=[previous_node_specs.name, conv_params.generate_param_name('conv', 'weights')],
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=[stride, stride],
            auto_pad='SAME_LOWER',
            name=layer_name
        )
        self._nodes.append(conv_node)
        # (Activation and batch normalization nodes are not needed here.)
        out_name = layer_name + '_bn' if batch_normalize else layer_name
        self.param_dict[layer_name] = conv_params
        return out_name, filters

    def _make_shortcut_node(self, layer_name, layer_dict):
        first_node_specs = self._get_previous_node_specs()
        second_node_specs = self._get_previous_node_specs(target_index=layer_dict['from'])
        channels = first_node_specs.channels
        shortcut_node = helper.make_node(
            'Add',
            inputs=[first_node_specs.name, second_node_specs.name],
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(shortcut_node)
        return layer_name, channels

    def _make_route_node(self, layer_name, layer_dict):
        route_node_indexes = layer_dict['layers']
        if len(route_node_indexes) == 1:
            if 'groups' in layer_dict.keys():
                groups = layer_dict['groups']
                group_id = int(layer_dict['group_id'])
                index = route_node_indexes[0]
                if index > 0:
                    index += 1
                route_node_specs = self._get_previous_node_specs(target_index=index)
                channels = route_node_specs.channels // groups
                out_name = layer_name + '_dummy'
            else:
                if route_node_indexes[0] < 0:
                    self.route_spec = route_node_indexes[0] - 1
                elif route_node_indexes[0] > 0:
                    self.route_spec = route_node_indexes[0] + 1
                out_name = layer_name + '_dummy'
                channels = 1
        else:
            inputs = []
            channels = 0
            for index in route_node_indexes:
                if index > 0:
                    index += 1
                route_node_specs = self._get_previous_node_specs(target_index=index)
                inputs.append(route_node_specs.name)
                channels += route_node_specs.channels
            route_node = helper.make_node(
                'Concat',
                axis=1,
                inputs=inputs,
                outputs=[layer_name],
                name=layer_name,
            )
            self._nodes.append(route_node)
            out_name = layer_name
        return out_name, channels

    def _make_upsample_node(self, layer_name, layer_dict):
        upsample_factor = float(layer_dict['stride'])
        scales = np.array([1.0, 1.0, upsample_factor, upsample_factor]).astype(np.float32)
        previous_node_specs = self._get_previous_node_specs()
        upsample_params = UpsampleParams(layer_name, scales)
        self.param_dict[layer_name] = upsample_params
        return layer_name, previous_node_specs.channels

    def _make_maxpool_node(self, layer_name, layer_dict):
        stride = layer_dict['stride']
        kernel_size = layer_dict['size']
        previous_node_specs = self._get_previous_node_specs()
        maxpool_node = helper.make_node(
            'MaxPool',
            inputs=[previous_node_specs.name],
            outputs=[layer_name],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
            auto_pad='SAME_UPPER',
            name=layer_name,
        )
        self._nodes.append(maxpool_node)
        return layer_name, previous_node_specs.channels

    def _make_yolo_node(self, layer_name, layer_dict):
        # Dummy yolo node (no weights)
        return layer_name + '_dummy', 1

# === REVERSE CONVERSION FUNCTION ===

def convert_onnx_to_darknet(cfg_path, onnx_path, output_path):
    # Parse Darknet cfg to recover layer ordering
    parser = DarkNetParser()
    layer_configs = parser.parse_cfg_file(cfg_path)

    # Determine output tensor information (used to initialize graph builder)
    category_num = get_category_num(cfg_path)
    output_convs = get_output_convs(layer_configs)
    c = (category_num + 5) * 3
    h, w = get_h_and_w(layer_configs)
    output_tensor_shapes = [
        [c, h // 8, w // 8],
        [c, h // 16, w // 16],
        [c, h // 32, w // 32],
        [c, h // 64, w // 64],
        [c, h // 128, w // 128]
    ]
    if len(output_convs) == 2:
        output_tensor_shapes = output_tensor_shapes[1:]
    output_tensor_shapes = output_tensor_shapes[:len(output_convs)]
    if not is_pan_arch(cfg_path):
        output_tensor_shapes.reverse()
    output_tensor_dims = OrderedDict(zip(output_convs, output_tensor_shapes))

    model_name = Path(onnx_path).stem

    # Build a dummy graph to populate the parameter dictionary (param_dict)
    builder = GraphBuilderONNX(model_name, output_tensor_dims, MAX_BATCH_SIZE)
    builder.build_dummy_graph(layer_configs)

    # Load the ONNX model and create a mapping from initializer names to numpy arrays.
    onnx_model = onnx.load(onnx_path)
    init_tensors = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    # Open the output Darknet weights file for binary writing.
    with open(output_path, "wb") as f:
        # Write Darknet header (5 int32 values; modify if needed)
        header = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        header.tofile(f)

        # Iterate over the parameters in the order they were created.
        for layer_name, param_obj in builder.param_dict.items():
            if isinstance(param_obj, ConvParams):
                if param_obj.batch_normalize:
                    # Write BN parameters: scale, bias, mean, var.
                    for suffix in ['scale', 'bias', 'mean', 'var']:
                        param_name = param_obj.generate_param_name('bn', suffix)
                        if param_name not in init_tensors:
                            raise ValueError(f"Initializer {param_name} not found in ONNX model")
                        weights = init_tensors[param_name].flatten().astype(np.float32)
                        weights.tofile(f)
                    # Write convolution weights.
                    param_name = param_obj.generate_param_name('conv', 'weights')
                    if param_name not in init_tensors:
                        raise ValueError(f"Initializer {param_name} not found in ONNX model")
                    weights = init_tensors[param_name].flatten().astype(np.float32)
                    weights.tofile(f)
                else:
                    # Without BN: write bias then convolution weights.
                    param_name = param_obj.generate_param_name('conv', 'bias')
                    if param_name not in init_tensors:
                        raise ValueError(f"Initializer {param_name} not found in ONNX model")
                    weights = init_tensors[param_name].flatten().astype(np.float32)
                    weights.tofile(f)
                    param_name = param_obj.generate_param_name('conv', 'weights')
                    if param_name not in init_tensors:
                        raise ValueError(f"Initializer {param_name} not found in ONNX model")
                    weights = init_tensors[param_name].flatten().astype(np.float32)
                    weights.tofile(f)
            elif isinstance(param_obj, UpsampleParams):
                # Darknet does not store upsample parameters.
                continue
            else:
                continue
    print("Darknet weights have been saved to:", output_path)

def main():
    args = parse_args()
    convert_onnx_to_darknet(args.config, args.onnx, args.output)

if __name__ == '__main__':
    main()

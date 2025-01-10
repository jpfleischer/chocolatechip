from cloudmesh.common.util import readfile
from glob import glob
from cloudmesh.common.console import Console
from pprint import pprint
import pyperclip
import yaml


def main(filename: str):


    # # Get the current directory
    # curdir = os.getcwd()

    # # Use glob to find all .txt files in the current directory
    # txt_files = glob(os.path.join(curdir, '*.yaml'))

    # if not txt_files:
    #     Console.error("No yaml files found in the current directory")
    #     quit()

    # my_text_files = []
    # for txt_file in txt_files:
    # read with yaml
    
    # if its not a yaml file, skip
    if not filename.endswith('.yaml'):
        Console.error("File is not a yaml file")
        return
    with open(filename, 'r') as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
        # my_text_files.append(content)

    legitimates = []

    # pprint(my_text_files)
    # for text in my_text_files:
    for key in content.keys():
        if content[key] == 'dangerous':
            if key.startswith('output_'):
                legitimates.append(key.split('_')[1])
            elif key.endswith('.mp4'):
                legitimates.append(key.split('_')[0])

    legitimates.sort()
    pprint(legitimates)

    pyperclip.copy(str(legitimates).replace('[', '').replace(']', ''))
                
    print('Now run unflag after cding to original made videos, not sprinkles')
                
    
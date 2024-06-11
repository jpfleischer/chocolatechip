import subprocess as sp
#from ultralytics import YOLOv10
import pandas as pd
import matplotlib as mpl

def get_input():
    print("Data Location: ")
    data = '/home/trey/yolov10/ultralytics/cfg/models/v10/yolov10s.yaml' #input()
    print("Model Location: ")
    model = '/home/trey/yolov10/obj.yaml' #input()
    print('Epochs: (default 100): ')
    epochs_input = input()
    epochs = int(epochs_input) if epochs_input else 100
    print('Image Size: (default 640): ')
    imSz_input = input()
    imSz = int(imSz_input) if imSz_input else 640

    return data, model, epochs, imSz


def run_command(cmd):
    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def main():
    
    #Getting all necessary parameters to run Ultralytics YOLOv10
    data, model, epochs, imSz = get_input()

    # format command for running ultralytics
    cmd = f"yolo detect train data={data} model={model} epochs={epochs} imgsz={imSz}"
    print(f'Running Ultralytics YOLOv10 with the following command: \n\t{cmd}')  

    # run command and capture output
    stdout, stderr = run_command(cmd)

    # check if stderr is empty, error occured while running command
    if stderr or stdout.empty():
        print(f'Error: {stderr}')
    
    '''
    # ALTERNATIVELY, run python command to train
    model = YOLOv10()
    model.train(data=data, epochs=epochs, imgsz=imSz)

    
    # read output, save results
    results = pd.read_csv('results.csv')

    print(results)
    '''

if __name__ == "__main__":
    main()
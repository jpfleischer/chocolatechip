import subprocess as sp
from ultralytics import YOLOv10
import pandas as pd
import matplotlib as mpl


data = '/home/trey/exp/yolov10/ultralytics/cfg/models/v10/yolov10s.yaml' #input()
print("Model Location: ")
model = '/home/trey/exp/yolov10/obj.yaml' #input()
epochs = 100
imSz = 640


model = YOLOv10()
model.train(data=data, epochs=epochs, imgsz=imSz)


# read output, save results
results = pd.read_csv('results.csv')

print(results)

# def run_command(cmd):
#     process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#     stdout, stderr = process.communicate()
#     return stdout.decode('utf-8'), stderr.decode('utf-8')

# def main():
    
#     #Getting all necessary parameters to run Ultralytics YOLOv10
#     data, model, epochs, imSz = get_input()

#     # format command for running ultralytics
#     cmd = f"yolo detect train data={data} model={model} epochs={epochs} imgsz={imSz}"
#     print(f'Running Ultralytics YOLOv10 with the following command: \n\t{cmd}')  

#     # run command and capture output
#     stdout, stderr = run_command(cmd)

#     # check if stderr is empty, error occured while running command
#     if stderr or stdout.empty():
#         print(f'Error: {stderr}')
    
#     '''
#     # ALTERNATIVELY, run python command to train

#     '''

# if __name__ == "__main__":
#     main()
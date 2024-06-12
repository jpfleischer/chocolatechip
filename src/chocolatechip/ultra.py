from ultralytics import YOLOv10
import pandas as pd
import matplotlib as plt

def main():
    data = '/home/trey/exp/yolov10/ultralytics/cfg/models/v10/yolov10s.yaml'
    epochs = 100
    imSz = 640

    # ALTERNATIVELY, run python command to train
    model = YOLOv10('yolov10s.pt')
    model.train(data=data, epochs=epochs, imgsz=imSz)

    # read output, save results
    results = pd.read_csv('results.csv')

    print(results)

    # plot results of loss / epochs
    plt.figure(10,6)
    plt.plot(results['epoch'], results['train/cls_om'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.grid(True)
    plt.show()

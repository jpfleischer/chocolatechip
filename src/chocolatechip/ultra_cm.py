from ultralytics import YOLOv10
import pandas as pd
import matplotlib.pyplot as plt
from cloudmesh.common.FlatDict import FlatDict

def main():
    config = FlatDict()

    config.load('config.yaml')
  
    data = 'obj.yaml'
    epochs = 100
    imSz = 640
    
    # run python command to train
    model = YOLOv10('yolov10s.pt')
    model.train(data=data, epochs=epochs, imgsz=imSz)

    results = pd.read_csv('~/yolov10/runs/detect/train14/results.csv') # needs to be changed depending on user and run count
    results.columns = results.columns.str.strip()    

    # plot results of loss / epochs
    plt.figure(figsize=(10, 6))
    plt.plot(results['epoch'], results['train/cls_om'])     # currently analyzes loss for clasification of the model
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.grid(True)
    
    plt.savefig('/home/trey/please/yolov10.png')    # needs to be changed depending on desired save location
    plt.show()
    

if __name__ == "__main__":
    main()

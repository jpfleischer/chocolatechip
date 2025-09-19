sept 16th

    To get latest ultralytics image, run:
    docker pull ultralytics/ultralytics:latest

    then run:
    docker run -it ultralytics/ultralytics:latest

    inside run:
    yolo --version

sept 18th 
    
    you now know how to jump into an ultralytics docker container.
    The Lego Gears dataset, meant for Darknet/YOLO exists at
    https://www.ccoderun.ca/programming/2024-05-01_LegoGears/
    
    I recommend that you unzip it on sunshine, do an ls on the folder, and examine the structure of the images and annotations. (vscode is helpful for exploring the images and file structure of the folder.) There should be someway for ultralytics to train on these files. You can explore the ultralytics documentation for training at https://docs.ultralytics.com/usage/cli/#train
    
    however, that command expects a yaml file, which is not standard for Darknet/YOLO.
    
    You have to

    figure out what information is within a yaml file that ultralytics (e.g. the yolo command) expects
    write that yaml file and try it til it works and starts training
    document the commands needed to replicate this training , as well as upload the yaml file to your ultralytics folder in semester-work.

    progress 
        - upload lego gears to chocolatechip/semester-work/fall2025/ultralytics/LegoGears (this is git ignored) 
        - you can also use lego gears 2, put in LegoGears_v2
        - made makefile that volume mounts this folder (might be useful for training)  
        - tried to make the yaml file, but having some problems with getting the dataset directory
        
        - attempted to change makefile to not require volume mounting of legogear1 2 and yaml but it didnt work (might try again later)
        - tried to run with model = yolov4.pt and didnt work, might have to load
        - got the error in latest_error.txt. It looks like it requires a cuda reinstallation on the docker image, which i dont know hwo to do
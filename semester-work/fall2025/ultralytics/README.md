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


# sep 26

the objective is to make sure that the training is equivalent
between ultralytics and darknet. this is done now that we know
we have 2134 epochs.

you need to benchmark ultralytics, by using similar logic that we did
in darknet. which is using StopWatch. the only thing you need to change
is running this command:

https://github.com/jpfleischer/chocolatechip/blob/b51d67508631722e266e58174fcf2993cf02bacf/semester-work/spring2025/darknet/LegoGearsFiles/run.py#L124-L129

instead of darknet detector, you are running `yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64`

and thats the only thing you should change, to give darknet and ultralytics
a fair comparison. we want to do benchmarking for ultralytics now. i want to see
how long it takes to run ultralytics training.

(ideally, and dont worry about this right now unless you feel like going above
and beyond, but we should not be copying code.
we should keep it DRY: dont repeat yourself, ideally you shouldnt copy that entire
run.py file, but just for getting it to work, its fine now. later on we will
incorporate this into a class or a chocolatechip import.)

so run ultralytics benchmark after copying the code from run.py (darknet),
and give us your benchmark file that is produced from ultralytics training.

progress
    - adjusted command
    - volume mount outputs to get outputs
    - error with cloudmesh when running
## extra: not necessary, but nice.

ideally, we need a python file that reads a Darknet cfg file.
it parses the parameters such as batch size, max_batches, steps(?)
and then runs the epochs_iterations.py file, which will then return
the correct ultralytics epoch count.

the intention is not to hardcode values.
im hardcoding values right now, in the Makefile. it says
```bash
yolo detect train data=LG_v2.yaml model=yolo11n.pt epochs=2134 batch=64
```

but i dont want to have to manually write 2134. your idea for the variable in
the makefile that gets output from the python script is good,
but be sure to add flags for that python script (something like
`EPOCH_SIZE=$(python3 epochs_iterations.py --batch_size --iterations)`)
would be good, but the question is, where does the batch_size come from?
you should parse the cfg with something like this, feel free to borrow logic
or ask GPT about it: https://github.com/jpfleischer/chocolatechip/blob/main/semester-work/spring2025/darknet/onnxconvert/cfgparser.py


# oct 8

When running on hipergator and pushing to github, ssh keys should be placed in ~/.ssh/authorized_keys 
instead of the classic location ~/.ssh/id_rsa.pub 
If this is not followed, you will be unable to login to hipergator and get the error "Permission denied (public key) "



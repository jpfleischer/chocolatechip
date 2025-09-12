# Apptainer

Please ensure your ssh key is set up with GitHub.
if not, do ssh-keygen and then once done do 

```bash
cat ~/.ssh/id_rsa.pub
```

and take that key and put it into your github settings new SSH key.


```bash
git clone git@github.com:jpfleischer/chocolatechip.git
cd chocolatechip/semester-work/spring2025/darknet
```
To run with gpu on HPC with slurm

```bash
make slurm
```

To run on a GPU with docker

```bash
make # to run with docker, may require:
# sudo usermod -aG docker $(whoami)
# then log back in.

make arun # to run with cpu, this is a bad idea
```

# Cars

To update the images from signal to HiPerGator

```bash
# this only gets the pictures that were resized from darkmark.
# our file quota is being reached so i am getting only what is necessary
rsync -av darkmark_image_cache/* j.fleischer@hpg.rc.ufl.edu:/blue/ranka/j.fleischer/annotation_data/darkmark_image_cache/.

# then the cfg and the other stuff. it gets the weights too which arent necessary .....
rsync -av --exclude='*/' ./ j.fleischer@hpg.rc.ufl.edu:/blue/ranka/j.fleischer/annotation_data/.
```

https://app.globus.org go here and log in
click Transfer or Sync to
In the left hand box put annotated_traffic_images
in the right hand box put UVA Standard Security Storage

uva's scratch, make  a folder named car_data

hit Select All in the left hand for annotated_traffic_images 
hit Start

On Afton,

```bash
cd /scratch/$USER/car_data
find . -type f -exec sed -i 's#/blue/ranka/j.fleischer/annotation_data/#/scratch/dje5dj/car_data/#g' {} +
find . -type f -exec sed -i 's#/data/ke_images_newrepo/darknet/data/combined/#/scratch/dje5dj/car_data/#g' {} +
```

# Other Notes

find stephanes discord server

https://github.com/hank-ai/darknet


and look for researchers, using the
dsicord search feature, who have created
a dockerfile for darknet.



i want you to build darknet in a docker container.
just build it.
output `darknet --version`


# 2/4

once darknet successfully compiles itself when you run that
.sh script, i want to see darknet version at the end of that
script.

once you verify that darknet works,

https://www.ccoderun.ca/programming/darknet_faq/

http://maltserver.cise.ufl.edu:6875/books/betos-book/page/legogears-confusion-matrix-yolov4-w-darknet

run the lego training in a docker shell (exec /bin/bash)  


### Steps for training Lego Model  

Ran the following code to train LegoGears_v2:  

```
# must clone the repo first
# git clone git@github.com:jpfleischer/chocolatechip.git
# cd chocolatechip/semester-work/spring2025/darknet
make
```



# 2/7

I have added the LegoGears.cfg file to darknet/ folder.
This is the one that we have pre-edited.

Then in the Dockerfile, we do 
`COPY LegoGears.cfg /workspace/LegoGears_v2/LegoGears.cfg`

So for the LegoGears.data, do the same thing,
changing /home/stephane to /workspace

((ideally it should all be a volume mount! which is done in the
makefile by going to docker run and doing --v ${CURDIR}:/workspace))

does darknet improve training, when it uses multiple GPUs? is it possible
to use multiple GPUs?


Use this command to validate model
darknet_03_display_videos detector demo annotations.data annotations.cfg annotations_best.weights -ext_output ~/07_2024-10-01_07-10-04.361.mp4

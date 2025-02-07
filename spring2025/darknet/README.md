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

Ran the following code to generate LegoGears_v2:  

```
wget https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip  
unzip legogears_2_dataset.zip  
rm legogears_2_dataset.zip  
python3 train_setup.py  
cat << EOF > LegoGears_v2/LegoGears.data  
classes = 5
train = /workspace/LegoGears_v2/LegoGears_train.txt
valid = /workspace/LegoGears_v2/LegoGears_valid.txt
names = /workspace/LegoGears_v2/LegoGears.names
backup = /workspace/LegoGears_v2
EOF
```

Next, edited LegoGears.cfg and changed the following:

`vim LegoGears_v2/LegoGears.cfg`  
set batch=64  
set subdivision=8

Build & Enter Docker Container:  
`make`  

Command to train in docker container:  
`darknet detector -map -dont_show -verbose -nocolor train /workspace/LegoGears_v2/LegoGears.data /workspace/LegoGears_v2/LegoGears.cfg 2>&1 | tee training_output.log`
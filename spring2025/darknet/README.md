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
# cd chocolatechip/spring2025/darknet
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

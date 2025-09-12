# use

you need the .env file that looks like 

```
CVAT_host=fillmeout
CVAT_port=fillmeout
CVAT_user=fillmeout
CVAT_passwd=fillmeout
```

then say `make`

# copy_annotations.py

right now the path is hardcoded but

sudo python3 copy_annotations.py

# other things


create a docker container that
installs cvat-api

after you install cvat-api,
you should be connecting to our cvat instance.
which is located at:
http://maltserver.cise.ufl.edu:8080/

you should pull the jobs that
use fisheye camera.

take a look at:
https://gitlab.com/maltgroup/custom-model/-/blob/main/main.py?ref_type=heads

https://gitlab.com/maltgroup/custom-model
https://docs.cvat.ai/docs/api_sdk/cli/



# 2/4

feedback:

great! we have cvatcli running.
now, we have to adapt
https://gitlab.com/maltgroup/custom-model/-/blob/main/main.py?ref_type=heads

dont use cloudmesh-shell,

just use subprocess.run, 
to run cvat-cli dump.

once you do that, youre gonna have all the images and
annotations in your docker container.

so, i suggest that you change your makefile
to do a volume mount.
-v ${CURDIR}:/app (or whatever your folder is in your docker container)
so you can see the images on the host. not just inside the docker container.



# 2/7

we have successfully identified the correct version number for cvat-cli
and we have identified the correct command for dumping with images.

moving forward, you are going to be augmenting your python script
to iterate through multiple tasks. not just 67 for example.
you have already done the volume mount, which is great, so that we dont
have to copy from the container to the host.

the python script, ideally, should parametrize the name, they cannot all
be named output.zip. maybe {task_number}.zip?

should be taking care of the unzipping automatically.

i want you to research, how can we have docker containers wait on each other?
because the workflow is:

1. run the cvat container, get all the images anad annotations, then exit.
2. once darknet sees that cvat is finished, its gonna want those image files. is there
a way for docker containers to "steal" each others files? or how to communicate those
files without introducing redundancy? we dont want a bunch of copies of all of those pictures,
its gonna add up to several hundred MB.

but first fix up the python script so it runs successfully.


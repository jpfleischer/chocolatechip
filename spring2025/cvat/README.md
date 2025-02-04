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



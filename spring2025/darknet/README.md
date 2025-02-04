find stephanes discord server

https://github.com/hank-ai/darknet


and look for researchers, using the
dsicord search feature, who have created
a dockerfile for darknet.



i want you to build darknet in a docker container.
just build it.
output `darknet --version`

### Steps:
Searched Discord and found linux users using docker image made by "sherenensberg"  
Searched his name and found a medium article they posted  
https://medium.com/@oschenberk/darknet-yolo-and-docker-02f52a927aba  

Followed docker link to get the image  
https://hub.docker.com/r/sherensberk/darknet/tags  

ran  
`sudo docker pull sherensberk/darknet:2204.550.1241-devel`  

Realized that this is an image, not a Dockerfile  
Tried recreating Dockerfile from https://hub.docker.com/layers/sherensberk/darknet/2204.550.1241-devel/images/sha256-1660b206cbbcb4e1a18d1c0fca22336f9513462c292eff40bd1e64ce8225abb0

This failed in the cuda installation process

Tried prompting ChatGPT to build it just from https://github.com/hank-ai/darknet instead  

Issue:
The docker image seemingly cannot access the NVIDIA GPU

Potential Fixes:

From: https://github.com/NVIDIA/nvidia-docker/issues/1033
create /etc/docker/daemon.json with this:
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
Then run sudo systemctl restart docker


Trying to set DOCKER_BUILDKIT=0 from
https://stackoverflow.com/questions/75641614/docker-run-has-access-to-gpu-but-docker-build-doesnt

This is the specific error for future reference:
CMake Error in src-lib/CMakeLists.txt:  
     CUDA_ARCHITECTURES is set to "native", but no GPU was detected.  

This user seems to have the same issue but was not able to resolve it:
https://github.com/hank-ai/darknet/issues/71

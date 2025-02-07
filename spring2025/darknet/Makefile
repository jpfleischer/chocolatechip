CONTAINER_NAME := my_darknet_container

build:
	@docker build -t darknet-container .
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) darknet-container

enter:
	@docker run --rm -it --gpus all --name $(CONTAINER_NAME) darknet-container

stop:
	@-docker stop $(CONTAINER_NAME)
	@-docker rm $(CONTAINER_NAME)

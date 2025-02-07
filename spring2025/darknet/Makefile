CONTAINER_NAME:="my_darknet_container"
build:
	@docker build -t darknet-container .
	@docker run --name=$(CONTAINER_NAME) --gpus all darknet-container

stop:
	@-docker stop $(CONTAINER_NAME)
	@-docker rm $(CONTAINER_NAME)
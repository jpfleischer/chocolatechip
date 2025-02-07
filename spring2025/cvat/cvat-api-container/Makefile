CONTAINER_NAME:="my_cvat_api"
HOST_DIR=$(CURDIR)
hello:
	docker build -t cvat-api .
	docker run --name=$(CONTAINER_NAME) -v $(HOST_DIR):/app cvat-api

stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
CONTAINER_NAME:="my_cvat_api"
hello:
	docker build -t cvat-api .
	docker run --name=$(CONTAINER_NAME) cvat-api

stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
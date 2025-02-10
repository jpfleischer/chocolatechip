CONTAINER_NAME:="my_cvat_api"
HOST_DIR=$(CURDIR)
hello: stop
	docker build -t cvat-api .
	docker run -d --name=$(CONTAINER_NAME) -v $(HOST_DIR):/app cvat-api

stop down:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash
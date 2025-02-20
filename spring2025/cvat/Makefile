CONTAINER_NAME:="my_cvat_api"
HOST_DIR=$(CURDIR)
hello: stop configure
	docker build -t cvat-api .
	docker run --rm --name=$(CONTAINER_NAME) --env-file .env -v $(HOST_DIR)/docker:/app cvat-api

stop down:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

configure:
	mkdir -p docker
	cp run_dump.py $(HOST_DIR)/docker/

shell: hello
	docker exec -it $(CONTAINER_NAME) /bin/bash
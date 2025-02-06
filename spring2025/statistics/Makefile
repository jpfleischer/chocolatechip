CONTAINER_NAME:=statsdocker

secrets:
	docker build --secret id=ACCESS_TOKEN,src=${CURDIR}/test.env -t ${CONTAINER_NAME} .  

normal:
	docker build -t ${CONTAINER_NAME} .

run:
	@echo "Running the docker container..."
	@docker run -e API_KEY="testkey" -v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip ${CONTAINER_NAME}

runt:
	@echo "Opening in terminal..."
	@docker run -it -v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip ${CONTAINER_NAME} /bin/bash
	
#@docker run -it -e API_KEY="testkey" -v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip ${CONTAINER_NAME} /bin/bash

stop:
	@-docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}




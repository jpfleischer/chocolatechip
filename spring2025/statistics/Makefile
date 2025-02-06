CONTAINER_NAME:=statsdocker

secrets:
	docker build -t ${CONTAINER_NAME} .  

normal:
	docker build -t ${CONTAINER_NAME} .

run:
	@echo "Running the docker container..."
	@docker run \
		-e API_KEY="testkey" -v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip\
		-v $(CURDIR)/test.env:/run/secrets/ACCESS_TOKEN:ro \
		${CONTAINER_NAME}

runt:
	@echo "Opening in terminal..."
	@docker run -it \
    	-v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip \
    	-v $(CURDIR)/test.env:/run/secrets/ACCESS_TOKEN:ro \
    	${CONTAINER_NAME} /bin/bash
	
# Once you make:
#   pip install -r requirements.txt
#   pip install -e . --config-settings editable_mode=strict in chocolatechip

stop:
	@-docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}

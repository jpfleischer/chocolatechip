CONTAINER_NAME:=statsdocker

secrets:
	@docker build \
		-t ${CONTAINER_NAME} .  

#		--secret id=ACCESS_TOKEN,src=default.env \

normal:
	docker build -t ${CONTAINER_NAME} .

run:
	@echo "Running the docker container..."
	@docker run \
		-e API_KEY="testkey" -v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip \
		--env-file test.env \
		${CONTAINER_NAME}

runt:
	@echo "Opening in terminal..."
	@docker run -it \
		-v $(CURDIR)/../../../chocolatechip:/usr/src/chocolatechip \
		--env-file test.env \
		${CONTAINER_NAME} /bin/bash -c "cd /usr/src/chocolatechip && \
		pip install -e . --config-settings editable_mode=strict && \
		exec /bin/bash"

# Once you make:
#	cd chocolatechip
#   pip install -r requirements.txt
#   pip install -e . --config-settings editable_mode=strict

stop:
	@-docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}

shell:
	docker run -it ${CONTAINER_NAME} /bin/bash

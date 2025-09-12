CONTAINER_NAME:=statsdocker

secrets:
	@echo "****************************************************"
	@echo "Building the docker container"
	@echo "****************************************************"
	@docker build \
		-t ${CONTAINER_NAME} .  

#		--secret id=ACCESS_TOKEN,src=default.env \

normal:
	docker build -t ${CONTAINER_NAME} .

run:
	@echo "****************************************************"
	@echo "Running the docker container"
	@echo "****************************************************"
	@docker run \
		-e API_KEY="testkey" -v $(CURDIR)/../../../../chocolatechip:/usr/src/chocolatechip \
		--env-file test.env \
		-e in_docker=true \
		${CONTAINER_NAME}

runt:
	@echo "****************************************************"
	@echo "Opening docker container in terminal"
	@echo "****************************************************"
	@docker run -it \
		-v $(CURDIR)/../../../../chocolatechip:/usr/src/chocolatechip \
		--env-file test.env \
		-e in_docker=true \
		${CONTAINER_NAME} /bin/bash -c "cd /usr/src/chocolatechip && \
		pip install -e . --config-settings editable_mode=strict && \
		exec /bin/bash"

master: secrets 
	@echo "****************************************************"
	@echo "Running heatmap.py"
	@echo "****************************************************"
	@docker run -it \
		-v $(CURDIR)/../../../../chocolatechip:/usr/src/chocolatechip \
		--env-file test.env \
		-e in_docker=true \
		${CONTAINER_NAME} /bin/bash -c "cd /usr/src/chocolatechip && \
		pip install -e . --config-settings editable_mode=strict && \
		cd /usr/src/chocolatechip/src/chocolatechip/heatmap && \
		python3 heatmap.py && \
		exec /bin/bash"

stop:
	@-docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}

shell:
	docker run -it ${CONTAINER_NAME} /bin/bash

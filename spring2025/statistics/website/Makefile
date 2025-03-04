CONTAINER_NAME:=flaskwebsite

build:
	@echo "****************************************************"
	@echo "Building the docker container"
	@echo "****************************************************"
	@docker build \
		-t ${CONTAINER_NAME} .  

runt:
	@docker run -it --rm -p 8000:8000 \
	-v $(CURDIR):/usr/src/app \
	--env-file test.env \
	${CONTAINER_NAME} /bin/bash -c \
	"cd app && gunicorn -w 4 -b 0.0.0.0:8000 server:app"

runtest:
	@docker run -it --rm -p 8000:8000 \
	-v $(CURDIR):/usr/src/app \
	--env-file test.env \
	${CONTAINER_NAME} /bin/bash

run:
	@docker run -it --rm -p 8000:8000 \
	-v $(CURDIR):/usr/src/app \
	${CONTAINER_NAME} /bin/bash

#gunicorn -w 4 -b 0.0.0.0:8000 server:app

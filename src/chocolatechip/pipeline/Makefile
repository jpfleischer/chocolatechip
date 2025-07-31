# Makefile for MySQL schema container with persistent volume

# --- Configurable parameters ---
IMAGE_NAME      := mysql-schema-image
CONTAINER_NAME  := mysqldb
VOLUME_NAME     := mysqldb-data
PORT            := 3306
DB_NAME         := testdb
ROOT_PASSWORD   := hereismymysqlpassword

# --- Targets ---
.PHONY: run stop down shell logs clean reset

run: stop
	docker build -t $(IMAGE_NAME) .
	docker run --name $(CONTAINER_NAME) \
		-v $(VOLUME_NAME):/var/lib/mysql \
		-e MYSQL_ROOT_PASSWORD=$(ROOT_PASSWORD) \
		-e MYSQL_DATABASE=$(DB_NAME) \
		-p $(PORT):3306 \
		-d $(IMAGE_NAME)

stop down:
	-docker stop $(CONTAINER_NAME)
	-docker rm   $(CONTAINER_NAME)

shell:
	docker exec -it $(CONTAINER_NAME) mysql -uroot -p$(ROOT_PASSWORD)

logs:
	docker logs $(CONTAINER_NAME)

clean reset: stop
	docker volume rm $(VOLUME_NAME)

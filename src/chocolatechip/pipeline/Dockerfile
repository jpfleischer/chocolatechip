# Dockerfile
FROM mysql:8.0

# these ENV vars will be used by the official image entrypoint
ENV MYSQL_ROOT_PASSWORD=hereismymysqlpassword
ENV MYSQL_DATABASE=testdb

# Copy your schema into the init directory
# Any *.sql or *.sh here is executed in alphabetical order
COPY schema.sql /docker-entrypoint-initdb.d/01-schema.sql

# Expose the MySQL port
EXPOSE 3306

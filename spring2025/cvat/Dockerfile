# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies for CVAT API
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    build-essential \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN pip install cvat-cli==2.12.1

# # Clone CVAT repository and install its dependencies
# RUN git clone https://github.com/cvat-ai/cvat /app/cvat
# WORKDIR /app/cvat
# RUN pip install -e .

# # Set environment variables for CVAT connection (you'll modify this later if needed)
# ENV CVAT_HOST=http://maltserver.cise.ufl.edu:8080/

# Expose necessary ports
EXPOSE 8080

# Command to start the application
# CMD ["cvat-cli", "--version"]
WORKDIR /app
CMD ["python", "/app/run_dump.py"]

# CMD ["sleep", "2000"]

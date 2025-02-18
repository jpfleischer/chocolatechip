import os

print("Hello World")

Docker_Check = os.environ.get('in_docker', False)

if Docker_Check:
    print('I am running in a Docker container')
else:
    print('I am not running in a docker container')


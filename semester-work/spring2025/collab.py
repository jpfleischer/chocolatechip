import subprocess

r = subprocess.call('cd cvat && make', shell=True,  text=True)

# TODO: add error detection for cvat setup

print('#'* 90)
print('Darknet Training')
print('#'* 90)


r2 = subprocess.call('cd darknet && make collab', shell=True,  text=True)


import subprocess

r = subprocess.call('cd cvat && make', shell=True,  text=True)

# there should be logic
# to see if r (the object that is returned by subprocess)
# has "unzipped"
#or, better, a sentinel
# "ALL TASKS COMPLETE!!!"
# if "ALL TASKS COMPLETE!!!" in r:
#    proceed.....
# else:
#    epic fail


print('#'* 90)
print('now for darknet')
print('#'* 90)


r2 = subprocess.call('cd darknet && make collab', shell=True,  text=True)


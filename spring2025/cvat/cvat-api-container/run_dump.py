import subprocess

subprocess.run([
    "cvat-cli",
    "--server-host", "maltserver.cise.ufl.edu",
    "--auth", "***REDACTED***:***REDACTED***",
    "--server-port", "8080",
    "ls"
])

# ls works, but 'dump'ing the task is not working for some reason
'''subprocess.run([
    "cvat-cli",
    "--server-host", "maltserver.cise.ufl.edu",
    "--auth", "***REDACTED***:***REDACTED***",
    "--server-port", "8080",
    "dump",
    "--format", "YOLO 1.1",
    "--with-images", "True",
    "67",
    "/app/tasks/67.zip"
])'''
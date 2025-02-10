
import subprocess

subprocess.run([
    "cvat-cli",
    "--server-host", "maltserver.cise.ufl.edu",
    "--auth", "REMOVED",
    "--server-port", "8080",
    "dump",
    "--format", "YOLO 1.1",
    "--with-images", "True",
    "67",
    "/app/tasks/67.zip"
])

# ls works, but 'dump'ing the task is not working for some reason
'''subprocess.run([
    "cvat-cli",
    "--server-host", "maltserver.cise.ufl.edu",
    "--auth", "REMOVED",
    "--server-port", "8080",
    "dump",
    "--format", "YOLO 1.1",
    "--with-images", "True",
    "67",
    "/app/tasks/67.zip"
])'''
from cloudmesh.common.StopWatch import StopWatch
import subprocess

if __name__ == "__main__":
    subprocess.call("python3 train_setup.py", shell=True)

    StopWatch.start("benchmark")
    subprocess.call("darknet detector -map -dont_show -verbose -nocolor train /workspace/LegoGears_v2/LegoGears.data /workspace/LegoGears_v2/LegoGears.cfg 2>&1 | tee training_output.log", shell=True)
    StopWatch.stop("benchmark")

    StopWatch.benchmark(filename="benchmark.txt")

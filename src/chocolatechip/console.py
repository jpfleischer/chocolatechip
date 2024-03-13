import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import readfile, writefile, path_expand
from docopt import docopt
from chocolatechip import Benchmark, Pipeline

def main():
    doc = """
chocolate chip. yum

Usage:
    chip fastmot
    chip pipeline
    chip help


Commands:
    fastmot   benchmark fastmot
    pipeline  restart the pipeline
    help      show this help message
    """

    if len(sys.argv) < 2 or sys.argv[1] in ['help', 'hello', 'hi']:
        print(doc)
        return

    args = docopt(doc, version='1.0')

    if args['fastmot']:
        Benchmark.main()

    if args['pipeline']:
        Pipeline.main()


if __name__ == "__main__":
    main()
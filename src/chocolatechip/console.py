import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import readfile, writefile, path_expand
from docopt import docopt
from chocolatechip import Benchmark, Pipeline, Stream, fastmotstarter, darknet


def main():
    doc = """
chocolate chip. yum

Usage:
    chip fastmot
    chip stream
    chip pipeline
    chip help
    chip stop
    chip down
    chip parallel
    chip darknet

Commands:
    fastmot   benchmark fastmot
    stream    benchmark max streams
    pipeline  restart the pipeline
    help      show this help message
    stop      stop all docker containers
    down      stop all docker containers
    parallel  start the parallel pipeline by starting fastmot
    darknet   enables darknet
    """

    if len(sys.argv) < 2 or sys.argv[1] in ['help', 'hello', 'hi']:
        print(doc)
        return

    args = docopt(doc, version='1.0')

    if args['fastmot']:
        Benchmark.main()

    if args['stream']:
        Stream.main()

    if args['pipeline']:
        Pipeline.main()

    if args['stop'] or args['down']:
        Pipeline.stop_everything()

    if args['parallel']:
        fastmotstarter.main()

    if args['darknet']:
        darknet.main()


if __name__ == "__main__":
    main()
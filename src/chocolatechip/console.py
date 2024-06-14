import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import readfile, writefile, path_expand
from docopt import docopt
from chocolatechip import Benchmark, Pipeline, Stream, fastmotstarter, latency, unflag
from chocolatechip.sprinkles import main as sprinkles


def main():
    doc = """
chocolate chip. yum

Usage:
    chip benchmark
    chip stream
    chip pipeline
    chip help
    chip stop
    chip down
    chip parallel
    chip up
    chip plain
    chip latency
    chip unflag
    chip sprinkles

Commands:
    benchmark  benchmark fastmot
    stream     benchmark max streams
    pipeline   restart the pipeline
    up         restart the pipeline
    help       show this help message
    stop       stop all docker containers
    down       stop all docker containers
    parallel   start the parallel pipeline by starting fastmot
    plain      normal pipeline without parallel
    latency    benchmark latency
    unflag     unflag conflicts that are invalid
    sprinkles  initiate unflagging gui
    """

    if len(sys.argv) < 2 or sys.argv[1] in ['help', 'hello', 'hi']:
        print(doc)
        return

    args = docopt(doc, version='1.0')

    if args['benchmark']:
        Benchmark.main()

    if args['stream']:
        Stream.main()

    if args['pipeline'] or args['up']:
        Pipeline.main()

    if args['stop'] or args['down']:
        Pipeline.stop_everything()

    if args['parallel']:
        fastmotstarter.main()

    if args['plain']:
        Pipeline.plain()

    if args['latency']:
        latency.main()

    if args['unflag']:
        unflag.main()

    if args['sprinkles']:
        
        sprinkles.main()
        # print(sprinkles)


if __name__ == "__main__":
    main()
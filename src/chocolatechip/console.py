import os
import sys
from cloudmesh.common.console import Console
from docopt import docopt
# from chocolatechip import Benchmark, Pipeline, Stream, fastmotstarter, latency
# from chocolatechip.unflag import main as unflag
# from chocolatechip.sprinkles import main as sprinkles


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
    chip offline
    chip latency
    chip unflag [<flags.yaml>]
    chip sprinkles
    chip sprinklesgui
    chip cvatzip
    chip extract <filename> <timestamp> [<frames>]

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
    offline    start the pipeline in offline mode with stored mp4 files
    latency    benchmark latency
    unflag     unflag conflicts that are invalid
    sprinkles  initiate automated moviepy
    sprinklesgui    initiate gui
    cvatzip    while standing in darknet folder, zip annotations for cvat
    extract        extract frames from a video file (requires: filename, timestamp, [frames])
    """

    if len(sys.argv) < 2 or sys.argv[1] in ['help', 'hello', 'hi']:
        print(doc)
        return

    args = docopt(doc, version='1.0')

    if args['benchmark']:
        from chocolatechip import Benchmark
        Benchmark.main()

    if args['stream']:
        from chocolatechip import Stream
        Stream.main()

    if args['pipeline'] or args['up']:
        from chocolatechip import Pipeline
        Pipeline.main()

    if args['stop'] or args['down']:
        from chocolatechip import Pipeline
        Pipeline.stop_everything()

    if args['parallel']:
        from chocolatechip import fastmotstarter
        fastmotstarter.main()

    if args['plain']:
        from chocolatechip import Pipeline
        Pipeline.plain()

    if args['offline']:
        from chocolatechip import Pipeline
        Pipeline.offline()

    if args['latency']:
        from chocolatechip import latency
        latency.main()

    if args['unflag']:
        yaml_file = args['<flags.yaml>'] or "flags.yaml"
        from chocolatechip.unflag import main as unflag_main
        unflag_main.main(yaml_file)

    if args['sprinkles']:
        from chocolatechip.sprinkles import main as sprinkles
        sprinkles.main()
        # print(sprinkles)

    if args['sprinklesgui']:
        from chocolatechip.sprinkles import gui as sprinkles_gui
        sprinkles_gui.main()

    if args['cvatzip']:
        from chocolatechip.cvat import cvatzip
        cvatzip.cvatzip()
        
    if args['extract']:
        from chocolatechip.cvat import extract
        extract.main()


if __name__ == "__main__":
    main()
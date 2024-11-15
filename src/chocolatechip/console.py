import os
import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import readfile, writefile, path_expand
from docopt import docopt
from chocolatechip import Benchmark, Pipeline, Stream, fastmotstarter, latency
from chocolatechip.unflag import main as unflag
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
    chip offline
    chip latency
    chip unflag
    chip sprinkles
    chip sprinklesgui
    chip eat <filename>

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
    eat    eat sprinklesgui yamls 
    peanuts   calendar gui to visualize data gathered by day
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

    if args['offline']:
        Pipeline.offline()

    if args['latency']:
        latency.main()

    if args['unflag']:
        unflag.main()

    if args['sprinkles']:
        
        sprinkles.main()
        # print(sprinkles)

    if args['sprinklesgui']:
        from chocolatechip.sprinkles import gui as sprinkles_gui
        sprinkles_gui.main()

    if args['eat']:
        from chocolatechip.unflag import eat
        if len(sys.argv) > 2 and sys.argv[2]:
            filename = sys.argv[2]
            if not os.path.isabs(filename):
                filename = os.path.abspath(filename)
            eat.main(filename)
        else:
            Console.error("Please provide a filename")
            return
    if args['peanuts']:
        from chocolatechip.peanuts.gui import main as peanuts
        peanuts()
    if args['clip']:
        if sys.argc < 4:
            print("Usage: clip <input_veh.csv> <input_ped.csv>")
            return
        from chocolatechip.clip import main as clip
        clip(sys.argv[2], sys.argv[3])

if __name__ == "__main__":
    main()
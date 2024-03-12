import sys
from cloudmesh.common.console import Console
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import readfile, writefile, path_expand
from docopt import docopt
import subprocess

def main():
    doc = """
chocolate chip. yum

Usage:
    chip fastmot
    chip help

Commands:
    fastmot   benchmark fastmot
    help      show this help message
    """

    if len(sys.argv) < 2 or sys.argv[1] in ['help', 'hello', 'hi']:
        print(doc)
        return

    args = docopt(doc, version='1.0')

    if args['fastmot']:
        print('wow, nice!')


if __name__ == "__main__":
    main()
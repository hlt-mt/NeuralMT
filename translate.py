import sys
import argparse
import codecs

from nmmt import NMTDecoder

import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


# Base models and Decoder definitions
# ======================================================================================================================
class Suggestion:
    def __init__(self, source, target, score):
        self.source = source
        self.target = target
        self.score = score

def addone(f):
    for line in f:
        yield line
    yield None

def run_main():
    # Args parse
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Run a forever-loop serving translation requests')
    parser.add_argument('model', metavar='MODEL', help='the path to the decoder model')
    parser.add_argument('-l', '--log-level', dest='log_level', metavar='LEVEL', help='select the log level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'], default='info')
    parser.add_argument('-g', '--gpu', type=int, dest='gpu', metavar='GPU', help='the index of the GPU to use',
                        default=None)
    parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')

    args = parser.parse_args()


    decoder = NMTDecoder(args.model, gpu_id=args.gpu, random_seed=3435)

    suggestions = None

    for line in addone(codecs.open(args.src, 'r', 'utf-8')):
        if line is not None:
            translation = decoder.translate(line, suggestions)
            print "%s" % translation

if __name__ == '__main__':
    run_main()
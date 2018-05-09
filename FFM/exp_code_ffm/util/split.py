#!/usr/bin/env python3

import argparse, sys, random

parser = argparse.ArgumentParser()
parser.add_argument('ratio', type=float)
parser.add_argument('src', type=str)
parser.add_argument('tr', type=str)
parser.add_argument('va', type=str)
ARGS = vars(parser.parse_args())

random.seed(0)

with open(ARGS['tr'], 'w') as f_tr, open(ARGS['va'], 'w') as f_va:
    for line in open(ARGS['src']):

        if random.random() < ARGS['ratio']:
            f_va.write(line)
        else:
            f_tr.write(line)

#!/usr/bin/env python3

import argparse, sys, math

parser = argparse.ArgumentParser()
parser.add_argument('log', type=str)
ARGS = vars(parser.parse_args())

best_loss, iter, best_iter = 100, 0, 1
for line in open(ARGS['log']):
    try:
        tokens = line.split()
        loss = float(tokens[2])
        iter += 1
        best_loss = loss
        best_iter = iter
    except:
        continue

print(best_loss, best_iter)

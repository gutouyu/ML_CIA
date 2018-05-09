#!/usr/bin/env python3

import argparse, sys, random

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
ARGS = vars(parser.parse_args())

map = {}

random.seed(0)

set0 = set()
set4 = set()

with open(ARGS['dst'], 'w') as f_dst:
    pos, neg = 0, 0
    for line in open(ARGS['src']):
        tokens = line.strip().split('\t')
        label, feats = tokens[0], tokens[1:]

        if label == '0':
            if random.random() < 0.9:
                continue
            output = '0'
            neg += 1
        else:
            output = '1'
            pos += 1

        for field, feat in enumerate(feats):
            if field == 0:
                feat = int(feat)
                if feat > 9:
                    feat = 10
                set0.add(feat)
            if field == 4:
                set4.add(feat)
            feat = '{0},{1}'.format(field, feat)
            if feat not in map:
                map[feat] = len(map) + 1
            dim = map[feat]
            output += ' {0}:{1}:1'.format(field, dim)

        f_dst.write(output + '\n')

print('num_instance = {0}'.format(pos + neg))
print('pos = {0}'.format(pos))
print('neg = {0}'.format(neg))
print('num_feats = {0}'.format(len(map)))
print(len(set0))
print(len(set4))

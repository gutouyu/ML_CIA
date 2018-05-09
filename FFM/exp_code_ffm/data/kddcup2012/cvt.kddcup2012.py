#!/usr/bin/env python3

import argparse, sys, random, math

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
parser.add_argument('--full', action='store_true')
ARGS = vars(parser.parse_args())

map = {}

random.seed(0)

set0 = set()
set4 = set()

with open(ARGS['dst']+'.ffm', 'w') as f_ffm, open(ARGS['dst']+'.svm', 'w') as f_svm, open('selected.txt', 'w') as selected:
    pos, neg = 0, 0
    for cur, line in enumerate(open(ARGS['src'])):
        tokens = line.strip().split('\t')
        label, feats = tokens[0], tokens[1:]

        if label == '0':
            if not ARGS['full'] and random.random() < 0.9:
                continue
            output = '0'
            neg += 1
        else:
            output = '1'
            pos += 1

        if not ARGS['full']:
            selected.write('{}\n'.format(cur+1))

        j_list = []
        output_svm = output
        val = 1.0/math.sqrt(len(feats))
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
            j_list.append(dim)
            output += ' {0}:{1}:{2:.5f}'.format(field, dim, val)

        for j in sorted(j_list):
            output_svm += ' {0}:{1:.5f}'.format(j, val)

        f_ffm.write(output + '\n')
        f_svm.write(output_svm + '\n')

print('num_instance = {0}'.format(pos + neg))
print('pos = {0}'.format(pos))
print('neg = {0}'.format(neg))
print('num_feats = {0}'.format(len(map)))

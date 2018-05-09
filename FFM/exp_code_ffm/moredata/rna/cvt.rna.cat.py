#!/usr/bin/env python3

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('tr_src', type=str)
parser.add_argument('te_src', type=str)
parser.add_argument('tr_dst', type=str)
parser.add_argument('te_dst', type=str)
ARGS = vars(parser.parse_args())

map = {}
def cvt(src_path, dst_path):
    with open(dst_path, 'w') as f_dst:
        for line in open(src_path):
            tokens = line.strip().split()
            label = tokens[0]
            if label == '1':
                output = '1'
            else:
                output = '-1'
            for field, token in enumerate(tokens[1:]):
                dim, val = token.split(':')
                feat = round(float(val), 3)
                feat = "{0}:{1}".format(field,feat)
                if feat not in map:
                    map[feat] = len(map) + 1
                output += ' {0}:{1}:1'.format(field, map[feat])
            f_dst.write(output + '\n')

cvt(ARGS['tr_src'], ARGS['tr_dst'])
cvt(ARGS['te_src'], ARGS['te_dst'])
print(len(map))

#!/usr/bin/env python3

import argparse, sys, math

parser = argparse.ArgumentParser()
parser.add_argument('tr_src', type=str)
parser.add_argument('va_src', type=str)
parser.add_argument('te_src', type=str)
parser.add_argument('tr_dst', type=str)
parser.add_argument('va_dst', type=str)
parser.add_argument('te_dst', type=str)
ARGS = vars(parser.parse_args())

map = {}
def cvt(src_path, dst_path):
    with open(dst_path, 'w') as f_dst:
        for line in open(src_path):
            tokens = line.strip().split()
            label = tokens[0]
            if float(label) == 1:
                output = '1'
            else:
                output = '-1'
            for token in tokens[1:]:
                field, val = token.split(':')
                field = int(field)
                val = float(val)
                if field in [11, 12]:
                    val *= 100
                elif field in [13, 16, 17, 18, 22]:
                    val *= 1000
                elif field in [14, 15, 19, 20, 21]:
                    val *= 10000
                feat = str(field) + ":" + str(int(val))
                if feat not in map:
                    map[feat] = len(map) + 1
                output += ' {0}:{1}:1'.format(field, map[feat])
            f_dst.write(output + '\n')

cvt(ARGS['tr_src'], ARGS['tr_dst'])
cvt(ARGS['va_src'], ARGS['va_dst'])
cvt(ARGS['te_src'], ARGS['te_dst'])
print(len(map))

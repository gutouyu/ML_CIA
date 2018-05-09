#!/usr/bin/env python3

import argparse, sys, math, csv

parser = argparse.ArgumentParser()
parser.add_argument('tr_src', type=str)
parser.add_argument('te_src', type=str)
parser.add_argument('tr_dst', type=str)
parser.add_argument('te_dst', type=str)
ARGS = vars(parser.parse_args())

FIELDNAMES = ['age', 'sex', 'capital-loss', 'occupation', 'hours-per-week', 'race', 'workclass', 'education-num', 'native-country', 'marital-status', 'education', 'relationship', 'capital-gain', 'fnlwgt']

map = {}
def cvt(src_path, dst_path):
    with open(dst_path, 'w') as f_dst:
        for row in csv.DictReader(open(src_path, 'r')):
            if row['income'].startswith('<=50K'):
                label = '0'
            else:
                label = '1'
            
            output = label
            for field, fieldname in enumerate(FIELDNAMES):
                feat = row[fieldname]
                if fieldname == 'age':
                    feat = int(feat) / 5
                elif fieldname == 'capital-loss':
                    if feat != '0' and feat != '?':
                        feat = 'log-' + str(int(math.log(float(feat))))
                elif fieldname == 'capital-gain':
                    if feat != '0' and feat != '?':
                        feat = 'log-' + str(int(math.log(float(feat))))
                elif fieldname == 'fnlwgt':
                    if feat != '0' and feat != '?':
                        feat = 'log-' + str(int(math.log(float(feat))))
                    
                feat = fieldname + '-' + str(feat)
                if feat not in map:
                    map[feat] = len(map)
                output += ' {0}:{1}:1'.format(field, map[feat])
            f_dst.write(output + '\n')

cvt(ARGS['tr_src'], ARGS['tr_dst'])
cvt(ARGS['te_src'], ARGS['te_dst'])
print(len(map))

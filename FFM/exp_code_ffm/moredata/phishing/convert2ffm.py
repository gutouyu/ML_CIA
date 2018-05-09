#!/usr/bin/env python

import os, sys, collections
import math

libsvm_path = sys.argv[1]

label_set = {}
num_feat = 0

feat_map = collections.defaultdict(int)
label_map = collections.defaultdict(int)

with open(libsvm_path + '.ffm', 'w') as out:
    for line in open(libsvm_path):
        line = line.strip().split()
        label = line[0]
        feats = line[1:]

        if label not in label_map:
            label_map[label] = len(label_map.keys())

        for index, feat in enumerate(feats):
            index, value = feat.split(':')

            if str(index) + '-' + str(value) not in feat_map:
                feat_map[str(index) + '-' + str(value)] = len(feat_map.keys())


    for line in open(libsvm_path):
        line = line.strip().split()
        label = line[0]
        feats = line[1:]

        out.write('{0}'.format(label_map[label]))

        for index, feat in enumerate(feats):
            index, value = feat.split(':')
            val = 1.0/math.sqrt(len(feats))
            out.write(' {0}:{1}:{2}'.format(index, feat_map[str(index) + '-' + str(value)], val))
        out.write('\n')

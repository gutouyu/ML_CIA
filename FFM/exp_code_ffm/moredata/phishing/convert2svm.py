#!/usr/bin/env python

import os, sys

src_path = sys.argv[1]

begin = 0
with open(src_path + '.svm', 'w') as out:
    for line in open(src_path):
        if 'data' in line:
            begin = 1
            continue
        if begin == 1:
            line = line.strip().split(',')

            label = line[-1]
            feats = line[:-1]

            out.write('{0}'.format(label))
            for index, feat in enumerate(feats):
                out.write(' {0}:{1}'.format(int(index) + 1, feat))
            out.write('\n')


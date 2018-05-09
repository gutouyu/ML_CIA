#!/usr/bin/env python

import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(output_path, 'w') as out:
    for line in open(input_path):
        line = line.strip().split()
        label = line[1]
        feats = line[2:]
        out.write('{0}'.format(label))

        for index, feat in enumerate(feats):
            out.write(' {0}:{1}:1'.format(index + 1, feat))
        out.write('\n')

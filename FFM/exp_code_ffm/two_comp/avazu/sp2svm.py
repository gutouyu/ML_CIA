#!/usr/bin/env python

import sys, math

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(output_path, 'w') as out:
    for line in open(input_path):
        line = line.strip().split()
        label = line[1]
        feats = list(set(map(float, line[2:])))

        val = 1.0/math.sqrt(len(feats))

        out.write('{0}'.format(label))

        for index, feat in enumerate(sorted(feats)):
            out.write(' {0}:{1}'.format(feat, val))
        out.write('\n')

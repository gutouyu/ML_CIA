import os
import sys
import math

input_path = sys.argv[1]
out_path = input_path + '.svm'

with open(out_path, 'w') as out:
    for line in open(input_path):
        line = line.strip().split()
        label = line[0]
        feats = line[1:]
        feat_list = [int(feat.split(':')[1]) for feat in feats]
        val = 1.0/math.sqrt(len(feat_list))

        out.write('{0}'.format(label))

        for feat in sorted(list(set(feat_list))):
            out.write(' {0}:{1:.5f}'.format(int(feat), val))
        out.write('\n')

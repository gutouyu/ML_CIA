#!/usr/bin/env python

import sys, random

src_path = sys.argv[1]
ratio = float(sys.argv[2])

random.seed(100)
with open(src_path + '.tr', 'w') as out_tr:
    with open(src_path + '.te', 'w') as out_te:
        for line in open(src_path):
            if random.random() > ratio:
                out_te.write(line)
            else:
                out_tr.write(line)

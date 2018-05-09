#!/usr/bin/env python3

import sys

if len(sys.argv) != 3:
    print('usage: {0} src_path dst_path'.format(sys.argv[0]))
    exit(1)

HEADER = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'

with open(sys.argv[2], 'w') as f:
    f.write(HEADER + '\n')
    for line in open(sys.argv[1]):
        line = line.replace(' ', '') 
        if line.strip() == '':
            continue
        f.write(line)

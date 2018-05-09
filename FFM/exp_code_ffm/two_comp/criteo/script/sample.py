#!/usr/bin/env python

import sys, random
random.seed(100)

input_path = sys.argv[1]
output_path = sys.argv[2]
ratio = float(sys.argv[3])

with open(output_path, 'w') as out:
    for index, line in enumerate(open(input_path)):
        if index == 0:
            out.write(line)
        else:
            if random.random() <= ratio:
                out.write(line)

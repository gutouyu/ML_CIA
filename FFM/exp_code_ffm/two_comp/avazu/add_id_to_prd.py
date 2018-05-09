#!/usr/bin/env python

import sys

file_with_id_path = sys.argv[1]
prd_path = sys.argv[2]
out_path = sys.argv[3]

with open(out_path, 'w') as out:
    out.write('id,click\n')

    for (line1, line2) in zip(open(file_with_id_path), open(prd_path)):
        line1 = line1.strip().split()
        line2 = line2.strip().split()
        prd_id = line1[0]
        prd = line2[0]
        out.write('{0},{1}\n'.format(prd_id, prd))

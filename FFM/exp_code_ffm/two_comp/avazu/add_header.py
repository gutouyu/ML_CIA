#!/usr/bin/env python
import sys

header_path = sys.argv[1]
va_path = sys.argv[2]
out_path = sys.argv[3]

with open(out_path, 'w') as out:
    header = open(header_path).readline()
    out.write(header)

    for line in open(va_path):
        out.write(line)

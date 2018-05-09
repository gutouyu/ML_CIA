#!/usr/bin/env python

import sys

id_path = sys.argv[1]
prediction_path = sys.argv[2]
output_path = sys.argv[3]

title = 'Id,Predicted'

with open(output_path, 'w') as out:
    out.write('{0}\n'.format(title))

    id_file = open(id_path)
    prediction = open(prediction_path)
    id_file.readline()

    for id_line, prediction in zip(id_file, prediction):
        ins_id = id_line.split(',')[0]
        out.write('{0},{1}'.format(ins_id, prediction))

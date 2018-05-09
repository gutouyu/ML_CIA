#!/usr/bin/env python

import os, sys, subprocess

def find_last_logloss(out):
    last_logloss = 10000
    for each_iter in out.split('\n')[2:-1]:
        logloss = float(each_iter.split()[-2])
        last_logloss = logloss
    return last_logloss

def find_min_logloss(out):
    min_val = 10000
    min_iter = 10000
    for each_iter in out.split('\n')[2:-1]:
        num_iter = float(each_iter.split()[0])
        logloss = float(each_iter.split()[-2])
        if logloss < min_val:
            min_val = logloss
            min_iter = num_iter
    return num_iter, min_val

l_list = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
k_list = [4, 8, 16]
r_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
t = 200

run_path = sys.argv[1]
tr_path = sys.argv[2]
te_path = sys.argv[3]

final_tr_path = sys.argv[4]
final_te_path = sys.argv[5]

output_path = sys.argv[6]

min_logloss = 10000
min_l = 0
min_k = 0
min_r = 0

for l in l_list:
    for k in k_list:
        for r in r_list:
            cmd = "./{run} -l {l} -s 12 -k {k} -r {r} -t {t} -p {te} {tr}".format(run=run_path, t = t, l = l, k = k, r = r, te = te_path, tr = tr_path)
            proc = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            out, err = proc.communicate()

            min_num_iter, min_logloss_in_out = find_min_logloss(out)

            if min_logloss_in_out < min_logloss:
                min_logloss = min_logloss_in_out
                min_l = l
                min_k = k
                min_r = r
                min_t = min_num_iter
            print 'oracle:\t', min_logloss, 'now:\t', min_logloss_in_out, 'paras:\t', l, k, r, min_t

cmd = "./{run} -l {l} -k {k} -r {r} -t {t} -p {te} {tr}".format(run=run_path, t = t, l = min_l, k = min_k, r = min_r, te = final_te_path, tr = final_tr_path)
print cmd
proc = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
out, err = proc.communicate()
min_logloss_in_out = find_last_logloss(out)

with open(output_path, 'w') as out:
    out.write('{0}\n'.format(min_logloss_in_out))

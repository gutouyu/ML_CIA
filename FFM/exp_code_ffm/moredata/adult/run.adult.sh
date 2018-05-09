#!/bin/bash

set -e

ln -sf ../../data/adult/adult.data
ln -sf ../../data/adult/adult.test
ln -sf ../../solvers/ffm_w_linear-train
ln -sf ../../solvers/fm_w_linear-train
ln -sf ../../solvers/linear-train
ln -sf ../../solvers/poly2_w_linear-train
ln -sf ../../util/best.py
ln -sf ../../util/last.py
ln -sf ../../util/split.py

sed 1d adult.test> adult.test_

./txt2csv.py adult.data adult.tr.csv
./txt2csv.py adult.test_ adult.te.csv

./cvt.adult.py adult.tr.csv adult.te.csv adult.trva.ffm adult.te.ffm

./split.py 0.2 adult.trva.ffm adult.tr.ffm adult.va.ffm

#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Linear, lambda = $lambda: "
#    ./linear-train -l $lambda -t 100 -p adult.va.ffm adult.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Poly2 w/ linear, lambda = $lambda: "
#    ./poly2_w_linear-train -l $lambda -t 50 -p adult.va.ffm adult.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0 0.0001 0.001 0.01; do 
#    for k in 40 100; do 
#        echo -n "FM w/ linear, lambda = $lambda, k = $k: "
#        ./fm_w_linear-train -k $k -l $lambda -t 30 -p adult.va.ffm adult.tr.ffm > log
#        ./best.py log
#    done
#done
#
#for lambda in 0 0.00001 0.0001; do 
#    for k in 4 8; do 
#        echo -n "FFM w/ linear, lambda = $lambda, k = $k: "
#        ./ffm_w_linear-train -k $k -l $lambda -t 30 -p adult.va.ffm adult.tr.ffm > log
#        ./best.py log
#    done
#done

echo "============test==============="

echo -n "Linear: "
./linear-train -l 0.00001 -t 97 -p adult.te.ffm adult.trva.ffm > log
./last.py log

echo -n "Poly2 w/ linear: "
./poly2_w_linear-train -l 0.000001 -t 12 -p adult.te.ffm adult.trva.ffm > log
./last.py log

echo -n "FM w/ linear: "
./fm_w_linear-train -k 40 -l 0.001 -t 10 -p adult.te.ffm adult.trva.ffm > log
./last.py log

echo -n "FFM w/ linear: "
./ffm_w_linear-train -k 8 -l 0.00001 -t 6 -p adult.te.ffm adult.trva.ffm > log
./last.py log

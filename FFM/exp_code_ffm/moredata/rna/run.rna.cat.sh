#!/bin/bash

set -e

#ln -sf ../../data/cod-rna
#ln -sf ../../data/cod-rna.t
#ln -sf ../../solvers/ffm_w_linear-train
#ln -sf ../../solvers/fm_w_linear-train
#ln -sf ../../solvers/linear-train
#ln -sf ../../solvers/poly2_w_linear-train
#ln -sf ../../util/best.py
#ln -sf ../../util/last.py
#ln -sf ../../util/split.py
#ln -sf ../../util/svm2ffm.py
#
#./cvt.rna.cat.py cod-rna cod-rna.t rna.trva.ffm rna.te.ffm
#
#./split.py 0.2 rna.trva.ffm rna.tr.ffm rna.va.ffm
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Linear, lambda = $lambda: "
#    ./linear-train -l $lambda -t 100 -r 2 -p rna.va.ffm rna.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Poly2 w/ linear, lambda = $lambda: "
#    ./poly2_w_linear-train -l $lambda -t 100 -r 2 -p rna.va.ffm rna.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    for k in 40 100; do 
#        echo -n "FM w/ linear, lambda = $lambda, k = $k: "
#        ./fm_w_linear-train -k $k -l $lambda -t 30 -p rna.va.ffm rna.tr.ffm > log
#        ./best.py log
#    done
#done
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    for k in 4 8; do 
#        echo -n "FFM w/ linear, lambda = $lambda, k = $k: "
#        ./ffm_w_linear-train -k $k -l $lambda -t 30 -p rna.va.ffm rna.tr.ffm > log
#        ./best.py log
#    done
#done

echo "============test==========="

echo -n "Linear: "
./linear-train -l 0 -t 100 -r 2 -p rna.te.ffm rna.trva.ffm > log
./last.py log

echo -n "Poly2 w/ linear: "
./poly2_w_linear-train -l 0 -r 2 -t 50 -p rna.te.ffm rna.trva.ffm > log
./last.py log

echo -n "FM w/ linear: "
./fm_w_linear-train -k 40 -l 0.0001 -t 15 -p rna.te.ffm rna.trva.ffm > log
./last.py log

echo -n "FFM w/ linear: "
./ffm_w_linear-train -k 8 -l 0.0001 -t 30 -p rna.te.ffm rna.trva.ffm > log
./last.py log

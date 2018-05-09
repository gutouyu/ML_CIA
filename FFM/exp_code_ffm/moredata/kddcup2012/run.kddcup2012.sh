#!/bin/bash

set -e

ln -sf ../../util/best.py
ln -sf ../../util/last.py
ln -sf ../../util/split.py
ln -sf ../../solvers/ffm_w_linear-train
ln -sf ../../solvers/fm_w_linear-train
ln -sf ../../solvers/linear-train
ln -sf ../../solvers/poly2_w_linear-train

ln -s ../../data/kddcup2012/kdd12.tr.ffm kddcup2012.tr.ffm
ln -s ../../data/kddcup2012/kdd12.va.ffm kddcup2012.va.ffm

#for lambda in 0.005 0.002 0.001 0.0005 0.0002 0.0001; do
#    for rate in 0.02 0.05 0.1 0.2; do
#       echo -n "Linear, lambda = $lambda, r = $rate: "
#       ./linear-train -s 24 -r $rate -l $lambda -t 50 -p kddcup2012.va.ffm kddcup2012.tr.ffm > log
#       ./best.py log
#    done
#done

#for lambda in 0.005 0.002 0.001 0.0005 0.0002 0.0001; do 
#    for rate in 0.02 0.05 0.1 0.2; do
#       echo -n "Poly2 w/ linear, lambda = $lambda, r = $rate: "
#       ./poly2_w_linear-train -s 24 -r $rate -l $lambda -t 50 -p kddcup2012.va.ffm kddcup2012.tr.ffm > log
#       ./best.py log
#    done
#done

#for k in 4 8 16 32 64; do
#    for lambda in 0.005 0.002 0.001 0.0005 0.0002 0.0001; do
#        for rate in 0.02 0.05 0.1 0.2; do
#           echo -n "FM w/ linear, lambda = $lambda, k = $k, r = $rate: "
#           ./fm_w_linear-train -s 24 -k $k -r $rate -l $lambda -t 50 -p kddcup2012.va.ffm kddcup2012.tr.ffm > log
#           ./best.py log
#        done
#    done
#done

#for k in 4 8 16; do
#    for lambda in 0.005 0.002 0.001 0.0005 0.0002 0.0001; do
#       for rate in 0.02 0.05 0.1 0.2; do
#           echo -n "FFM w/ linear, lambda = $lambda, k = $k, r = $rate: "
#           ./ffm_w_linear-train -s 24 -k $k -r $rate -l $lambda -t 50 -p kddcup2012.va.ffm kddcup2012.tr.ffm > log
#           ./best.py log
#       done
#    done
#done

#echo "============test=============="

ln -s ../../data/kddcup2012/kdd12.trva.ffm kddcup2012.trva.ffm
ln -s ../../data/kddcup2012/kdd12.te.ffm kddcup2012.te.ffm

echo -n "Linear: "
./linear-train -s 24 -r 0.2 -l 0.0001 -t 50 -p kddcup2012.te.ffm kddcup2012.trva.ffm > log
./last.py log

echo -n "Poly2 w/ linear: "
./poly2_w_linear-train -s 24 -r 0.2 -l 0.0001 -t 21 -p kddcup2012.te.ffm kddcup2012.trva.ffm > log
./last.py log

echo -n "FM w/ linear: "
./fm_w_linear-train -s 24 -k 16 -r 0.05 -l 0.0002 -t 16 -p kddcup2012.te.ffm kddcup2012.trva.ffm > log
./last.py log

echo -n "FFM w/ linear: "
./ffm_w_linear-train -s 24 -k 8 -r 0.05 -l 0.0001 -t 29 -p kddcup2012.te.ffm kddcup2012.trva.ffm > log
./last.py log

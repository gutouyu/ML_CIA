#!/bin/bash

set -e

#ln -sf ../../data/ijcnn1.tr
#ln -sf ../../data/ijcnn1.val
#ln -sf ../../data/ijcnn1.t
#ln -sf ../../solvers/ffm_w_linear-train
#ln -sf ../../solvers/fm_w_linear-train
#ln -sf ../../solvers/linear-train
#ln -sf ../../solvers/poly2_w_linear-train
#ln -sf ../../util/best.py
#ln -sf ../../util/last.py
#
#./cvt.ijcnn.cat.py ijcnn1.tr ijcnn1.val ijcnn1.t ijcnn.tr.ffm ijcnn.va.ffm ijcnn.te.ffm
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Linear, lambda = $lambda: "
#    ./linear-train -l $lambda -t 100 -p ijcnn.va.ffm ijcnn.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0 0.000001 0.00001 0.0001; do 
#    echo -n "Poly2 w/ linear, lambda = $lambda: "
#    ./poly2_w_linear-train -l $lambda -t 100 -p ijcnn.va.ffm ijcnn.tr.ffm > log
#    ./best.py log
#done
#
#for lambda in 0.001 0.01 0.1; do 
#    for k in 40 100; do 
#        echo -n "FM w/ linear, lambda = $lambda, k = $k: "
#        ./fm_w_linear-train -r 0.05 -k $k -l $lambda -t 30 -p ijcnn.va.ffm ijcnn.tr.ffm > log
#        ./best.py log
#    done
#done
#
#for lambda in 0 0.00001 0.0001 0.001 0.01; do 
#    for k in 4 8; do 
#        echo -n "FFM w/ linear, lambda = $lambda, k = $k: "
#        ./ffm_w_linear-train -k $k -l $lambda -t 30 -p ijcnn.va.ffm ijcnn.tr.ffm > log
#        ./best.py log
#    done
#done

echo "============test==============="

cp ijcnn.tr.ffm ijcnn.trva.ffm 
chmod +w ijcnn.trva.ffm 
cat ijcnn.va.ffm >> ijcnn.trva.ffm
echo -n "Linear: "
./linear-train -l 0 -t 27 -p ijcnn.te.ffm ijcnn.trva.ffm > log
./last.py log

echo -n "Poly2 w/ linear: "
./poly2_w_linear-train -l 0 -t 15 -p ijcnn.te.ffm ijcnn.trva.ffm > log
./last.py log

echo -n "FM w/ linear: "
./fm_w_linear-train -r 0.05 -k 100 -l 0.01 -t 9 -p ijcnn.te.ffm ijcnn.trva.ffm > log
./last.py log

echo -n "FFM w/ linear: "
./ffm_w_linear-train -k 4 -l 0.001 -t 15 -p ijcnn.te.ffm ijcnn.trva.ffm > log
./last.py log

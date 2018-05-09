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

./grid_para.py ffm_w_linear-train adult.tr.ffm adult.va.ffm adult.trva.ffm adult.te.ffm adult.ffm
./grid_para.py fm_w_linear-train adult.tr.ffm adult.va.ffm adult.trva.ffm adult.te.ffm adult.fm
./grid_para.py poly2_w_linear-train adult.tr.ffm adult.va.ffm adult.trva.ffm adult.te.ffm adult.poly2
./grid_para.py linear-train adult.tr.ffm adult.va.ffm adult.trva.ffm adult.te.ffm adult.linear

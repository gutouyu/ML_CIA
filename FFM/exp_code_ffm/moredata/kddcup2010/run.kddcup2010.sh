#!/bin/bash
set -e

ln -sf ../../solvers/ffm_w_linear-train
ln -sf ../../solvers/fm_w_linear-train
ln -sf ../../solvers/linear-train
ln -sf ../../solvers/poly2_w_linear-train

ln -sf ../../data/kddcup2010/kddb-raw.tr.tr.ffm kddcup2010.data.tr.tr.ffm
ln -sf ../../data/kddcup2010/kddb-raw.tr.te.ffm kddcup2010.data.tr.te.ffm
ln -sf ../../data/kddcup2010/kddb-raw.ffm kddcup2010.data.tr.ffm
ln -sf ../../data/kddcup2010/kddb-raw.t.ffm kddcup2010.data.te.ffm

./grid_para.py ffm_w_linear-train kddcup2010.data.tr.tr.ffm kddcup2010.data.tr.te.ffm   kddcup2010.data.tr.ffm kddcup2010.data.te.ffm kddcup2010.ffm
./grid_para.py fm_w_linear-train kddcup2010.data.tr.tr.ffm kddcup2010.data.tr.te.ffm   kddcup2010.data.tr.ffm kddcup2010.data.te.ffm kddcup2010.fm
./grid_para.py poly2_w_linear-train kddcup2010.data.tr.tr.ffm kddcup2010.data.tr.te.ffm   kddcup2010.data.tr.ffm kddcup2010.data.te.ffm kddcup2010.poly2
./grid_para.py linear-train kddcup2010.data.tr.tr.ffm kddcup2010.data.tr.te.ffm   kddcup2010.data.tr.ffm kddcup2010.data.te.ffm kddcup2010.linear

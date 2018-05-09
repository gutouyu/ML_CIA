#!/bin/bash

set -e

ln -sf ../../../solvers/ffm_w_linear-train
ln -sf ../../../solvers/fm_w_linear-train
ln -sf ../../../solvers/linear-train
ln -sf ../../../solvers/poly2_w_linear-train

./grid_para_ffm.py ffm_w_linear-train kddb-cj.tr.ffm kddb-cj.va.ffm kddb-cj.ffm kddb-cj.t.ffm kddb-cj.ffm
./grid_para.py fm_w_linear-train kddb-cj.tr.ffm kddb-cj.va.ffm kddb-cj.ffm kddb-cj.t.ffm kddb-cj.fm
./grid_para.py poly2_w_linear-train kddb-cj.tr.ffm kddb-cj.va.ffm kddb-cj.ffm kddb-cj.t.ffm kddb-cj.poly2
./grid_para.py linear-train kddb-cj.tr.ffm kddb-cj.va.ffm kddb-cj.ffm kddb-cj.t.ffm kddb-cj.linear

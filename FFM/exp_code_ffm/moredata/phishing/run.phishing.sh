#!/bin/bash

set -e

ln -sf ../../data/phishing/Training\ Dataset.arff ./phishing.data
ln -sf ../../solvers/ffm_w_linear-train
ln -sf ../../solvers/fm_w_linear-train
ln -sf ../../solvers/linear-train
ln -sf ../../solvers/poly2_w_linear-train

./convert2svm.py phishing.data
./convert2ffm.py phishing.data.svm
./split.py phishing.data.svm.ffm 0.8
./split.py phishing.data.svm.ffm.tr 0.8


echo "griding parameters for phishing"
./grid_para.py ffm_w_linear-train phishing.data.svm.ffm.tr.tr phishing.data.svm.ffm.tr.te  phishing.data.svm.ffm.tr phishing.data.svm.ffm.te phishing.ffm
./grid_para.py fm_w_linear-train phishing.data.svm.ffm.tr.tr phishing.data.svm.ffm.tr.te  phishing.data.svm.ffm.tr phishing.data.svm.ffm.te phishing.fm
./grid_para.py poly2_w_linear-train phishing.data.svm.ffm.tr.tr phishing.data.svm.ffm.tr.te  phishing.data.svm.ffm.tr phishing.data.svm.ffm.te phishing.poly2
./grid_para.py linear-train phishing.data.svm.ffm.tr.tr phishing.data.svm.ffm.tr.te  phishing.data.svm.ffm.tr phishing.data.svm.ffm.te phishing.linear

#!/bin/bash

git clone https://github.com/guestwalk/kaggle-avazu.git
cd kaggle-avazu
make

gunzip ../../../data/avazu/train.gz
gunzip ../../../data/avazu/test.gz

ln -sf ../../../data/avazu/train tr.r0.csv
./add_dummy_label.py ../../../data/avazu/test va.r0.csv

cd base
make -C mark/mark1 && ln -sf mark/mark1/mark1

./util/gen_data.py ../tr.r0.csv ../va.r0.csv tr.r0.app.new.csv va.r0.app.new.csv tr.r0.site.new.csv va.r0.site.new.csv
./util/parallelizer.py -s 12 ./converter/2.py tr.r0.app.new.csv va.r0.app.new.csv tr.r0.app.sp va.r0.app.sp
./util/parallelizer.py -s 12 ./converter/2.py tr.r0.site.new.csv va.r0.site.new.csv tr.r0.site.sp va.r0.site.sp

cd ..
cd ..

./sp2ffm.py kaggle-avazu/base/tr.r0.app.sp kaggle-avazu/base/tr.r0.app.sp.ffm
./sp2ffm.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.ffm
./sp2ffm.py kaggle-avazu/base/tr.r0.site.sp kaggle-avazu/base/tr.r0.site.sp.ffm
./sp2ffm.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.ffm

# train ffm
../../solvers/ffm-train -l 0.00002 -k 4 -t 4 -r 0.2 kaggle-avazu/base/tr.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.ffm.model > avazu.app.ffm.log
../../solvers/ffm-train -l 0.00002 -k 4 -t 4 -r 0.2 kaggle-avazu/base/tr.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.ffm.model > avazu.site.ffm.log

../../solvers/ffm-predict kaggle-avazu/base/va.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.ffm.model kaggle-avazu/base/va.r0.app.sp.ffm.prd.no.id
../../solvers/ffm-predict kaggle-avazu/base/va.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.ffm.model kaggle-avazu/base/va.r0.site.sp.ffm.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.ffm.prd.no.id kaggle-avazu/base/va.r0.app.sp.ffm.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.ffm.prd.no.id kaggle-avazu/base/va.r0.site.sp.ffm.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.ffm.prd va.r0.app.sp.ffm.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.ffm.prd va.r0.site.sp.ffm.prd.pickle

./util/merge_prediction.py va.r0.app.sp.ffm.prd.pickle va.r0.site.sp.ffm.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.ffm

# train fm
cd ..
cd ..
../../solvers/fm-train -l 0.00002 -k 40 -t 8 -r 0.05 kaggle-avazu/base/tr.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.fm.model > avazu.app.fm.1.log
../../solvers/fm-train -l 0.00002 -k 40 -t 8 -r 0.05 kaggle-avazu/base/tr.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.fm.model > avazu.site.fm.1.log

../../solvers/fm-predict kaggle-avazu/base/va.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.fm.model kaggle-avazu/base/va.r0.app.sp.fm.prd.no.id
../../solvers/fm-predict kaggle-avazu/base/va.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.fm.model kaggle-avazu/base/va.r0.site.sp.fm.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.fm.prd.no.id kaggle-avazu/base/va.r0.app.sp.fm.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.fm.prd.no.id kaggle-avazu/base/va.r0.site.sp.fm.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.fm.prd va.r0.app.sp.fm.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.fm.prd va.r0.site.sp.fm.prd.pickle

./util/merge_prediction.py va.r0.app.sp.fm.prd.pickle va.r0.site.sp.fm.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.fm.1

# train fm
cd ..
cd ..
../../solvers/fm-train -l 0.00002 -k 100 -t 8 -r 0.05 kaggle-avazu/base/tr.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.fm.model > avazu.app.fm.2.log
../../solvers/fm-train -l 0.00002 -k 100 -t 8 -r 0.05 kaggle-avazu/base/tr.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.fm.model > avazu.site.fm.2.log

../../solvers/fm-predict kaggle-avazu/base/va.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.fm.model kaggle-avazu/base/va.r0.app.sp.fm.prd.no.id
../../solvers/fm-predict kaggle-avazu/base/va.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.fm.model kaggle-avazu/base/va.r0.site.sp.fm.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.fm.prd.no.id kaggle-avazu/base/va.r0.app.sp.fm.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.fm.prd.no.id kaggle-avazu/base/va.r0.site.sp.fm.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.fm.prd va.r0.app.sp.fm.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.fm.prd va.r0.site.sp.fm.prd.pickle

./util/merge_prediction.py va.r0.app.sp.fm.prd.pickle va.r0.site.sp.fm.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.fm.2

# train poly2
cd ..
cd ..
../../solvers/poly2-train -l 0 -t 10 -r 0.2 kaggle-avazu/base/tr.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.poly2.model > avazu.app.poly2.log
../../solvers/poly2-train -l 0 -t 10 -r 0.2 kaggle-avazu/base/tr.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.poly2.model > avazu.site.poly2.log

../../solvers/poly2-predict kaggle-avazu/base/va.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.poly2.model kaggle-avazu/base/va.r0.app.sp.poly2.prd.no.id
../../solvers/poly2-predict kaggle-avazu/base/va.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.poly2.model kaggle-avazu/base/va.r0.site.sp.poly2.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.poly2.prd.no.id kaggle-avazu/base/va.r0.app.sp.poly2.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.poly2.prd.no.id kaggle-avazu/base/va.r0.site.sp.poly2.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.poly2.prd va.r0.app.sp.poly2.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.poly2.prd va.r0.site.sp.poly2.prd.pickle

./util/merge_prediction.py va.r0.app.sp.poly2.prd.pickle va.r0.site.sp.poly2.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.poly2


# train linear
cd ..
cd ..
../../solvers/linear-train -l 0 -t 13 -r 0.2 kaggle-avazu/base/tr.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.linear.model > avazu.app.linear.log
../../solvers/linear-train -l 0 -t 13 -r 0.2 kaggle-avazu/base/tr.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.linear.model > avazu.site.linear.log

../../solvers/linear-predict kaggle-avazu/base/va.r0.app.sp.ffm kaggle-avazu/base/tr.r0.app.sp.linear.model kaggle-avazu/base/va.r0.app.sp.linear.prd.no.id
../../solvers/linear-predict kaggle-avazu/base/va.r0.site.sp.ffm kaggle-avazu/base/tr.r0.site.sp.linear.model kaggle-avazu/base/va.r0.site.sp.linear.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.linear.prd.no.id kaggle-avazu/base/va.r0.app.sp.linear.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.linear.prd.no.id kaggle-avazu/base/va.r0.site.sp.linear.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.linear.prd va.r0.app.sp.linear.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.linear.prd va.r0.site.sp.linear.prd.pickle

./util/merge_prediction.py va.r0.app.sp.linear.prd.pickle va.r0.site.sp.linear.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.linear


# train poly2-hash-cd
cd ..
cd ..

./ffm2svm.py kaggle-avazu/base/tr.r0.app.sp.ffm 
./ffm2svm.py kaggle-avazu/base/tr.r0.site.sp.ffm
./ffm2svm.py kaggle-avazu/base/va.r0.app.sp.ffm 
./ffm2svm.py kaggle-avazu/base/va.r0.site.sp.ffm

../../solvers/liblr-poly2-hash/train-p2hash -s 7 -c 1 -n 24 -h 3 kaggle-avazu/base/tr.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.poly2.hash.cd.model > avazu.app.poly2.hash.cd.log
../../solvers/liblr-poly2-hash/train-p2hash -s 7 -c 1 -n 24 -h 3 kaggle-avazu/base/tr.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.poly2.hash.cd.model > avazu.site.poly2.hash.cd.log

../../solvers/liblr-poly2-hash/predict-p2hash -b 1 kaggle-avazu/base/va.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.poly2.hash.cd.model kaggle-avazu/base/va.r0.app.sp.poly2.hash.cd.prd.no.id.raw
../../solvers/liblr-poly2-hash/predict-p2hash -b 1 kaggle-avazu/base/va.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.poly2.hash.cd.model kaggle-avazu/base/va.r0.site.sp.poly2.hash.cd.prd.no.id.raw

cat kaggle-avazu/base/va.r0.app.sp.poly2.hash.cd.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.app.sp.poly2.hash.cd.prd.no.id
cat kaggle-avazu/base/va.r0.site.sp.poly2.hash.cd.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.site.sp.poly2.hash.cd.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.poly2.hash.cd.prd.no.id kaggle-avazu/base/va.r0.app.sp.poly2.hash.cd.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.poly2.hash.cd.prd.no.id kaggle-avazu/base/va.r0.site.sp.poly2.hash.cd.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.poly2.hash.cd.prd va.r0.app.sp.poly2.hash.cd.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.poly2.hash.cd.prd va.r0.site.sp.poly2.hash.cd.prd.pickle

./util/merge_prediction.py va.r0.app.sp.poly2.hash.cd.prd.pickle va.r0.site.sp.poly2.hash.cd.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.poly2.hash.cd

# train poly2-hash-newton
cd ..
cd ..
../../solvers/liblr-poly2-hash/train-p2hash -s 0 -c 1 kaggle-avazu/base/tr.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.poly2.hash.newton.model > avazu.app.poly2.hash.newton.log
../../solvers/liblr-poly2-hash/train-p2hash -s 0 -c 1 kaggle-avazu/base/tr.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.poly2.hash.newton.model > avazu.site.poly2.hash.newton.log

../../solvers/liblr-poly2-hash/predict-p2hash -b 1 kaggle-avazu/base/va.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.poly2.hash.newton.model kaggle-avazu/base/va.r0.app.sp.poly2.hash.newton.prd.no.id.raw
../../solvers/liblr-poly2-hash/predict-p2hash -b 1 kaggle-avazu/base/va.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.poly2.hash.newton.model kaggle-avazu/base/va.r0.site.sp.poly2.hash.newton.prd.no.id.raw

cat kaggle-avazu/base/va.r0.app.sp.poly2.hash.newton.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.app.sp.poly2.hash.newton.prd.no.id
cat kaggle-avazu/base/va.r0.site.sp.poly2.hash.newton.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.site.sp.poly2.hash.newton.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.poly2.hash.newton.prd.no.id kaggle-avazu/base/va.r0.app.sp.poly2.hash.newton.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.poly2.hash.newton.prd.no.id kaggle-avazu/base/va.r0.site.sp.poly2.hash.newton.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.poly2.hash.newton.prd va.r0.app.sp.poly2.hash.newton.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.poly2.hash.newton.prd va.r0.site.sp.poly2.hash.newton.prd.pickle

./util/merge_prediction.py va.r0.app.sp.poly2.hash.newton.prd.pickle va.r0.site.sp.poly2.hash.newton.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.poly2.hash.newton


cd ..
cd ..

# train libFM
dimension=1,1,40
iteration=20
method=als
regular=1,1,40

../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.app.sp.ffm.svm -train kaggle-avazu/base/tr.r0.app.sp.ffm.svm -out kaggle-avazu/base/avazu.app.libfm.1.prd.no.id > avazu.app.libfm.1.log
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.site.sp.ffm.svm -train kaggle-avazu/base/tr.r0.site.sp.ffm.svm -out kaggle-avazu/base/avazu.site.libfm.1.prd.no.id > avazu.site.libfm.1.log

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/avazu.app.libfm.1.prd.no.id kaggle-avazu/base/avazu.app.libfm.1.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/avazu.site.libfm.1.prd.no.id kaggle-avazu/base/avazu.site.libfm.1.prd
cd kaggle-avazu/base/

./util/pickle_prediction.py avazu.app.libfm.1.prd avazu.app.libfm.1.prd.pickle
./util/pickle_prediction.py avazu.site.libfm.1.prd avazu.site.libfm.1.prd.pickle

./util/merge_prediction.py avazu.app.libfm.1.prd.pickle avazu.site.libfm.1.prd.pickle va.r0.prd.libfm.1.pickle
./util/unpickle_prediction.py va.r0.prd.libfm.1.pickle base.r0.prd.libfm.1


cd ..
cd ..

# libfm model
dimension=1,1,40
iteration=50
method=als
regular=1,1,40

../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.app.sp.ffm.svm -train kaggle-avazu/base/tr.r0.app.sp.ffm.svm -out kaggle-avazu/base/avazu.app.libfm.2.prd.no.id > avazu.app.libfm.2.log
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.site.sp.ffm.svm -train kaggle-avazu/base/tr.r0.site.sp.ffm.svm -out kaggle-avazu/base/avazu.site.libfm.2.prd.no.id > avazu.site.libfm.2.log

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/avazu.app.libfm.2.prd.no.id kaggle-avazu/base/avazu.app.libfm.2.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/avazu.site.libfm.2.prd.no.id kaggle-avazu/base/avazu.site.libfm.2.prd
cd kaggle-avazu/base/

./util/pickle_prediction.py avazu.app.libfm.2.prd avazu.app.libfm.2.prd.pickle
./util/pickle_prediction.py avazu.site.libfm.2.prd avazu.site.libfm.2.prd.pickle

./util/merge_prediction.py avazu.app.libfm.2.prd.pickle avazu.site.libfm.2.prd.pickle va.r0.prd.libfm.2.pickle
./util/unpickle_prediction.py va.r0.prd.libfm.2.pickle base.r0.prd.libfm.2

cd ..
cd ..

# libfm model
dimension=1,1,40
iteration=50
method=als
regular=1,1,40

../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.app.sp.ffm.svm -train kaggle-avazu/base/tr.r0.app.sp.ffm.svm -out kaggle-avazu/base/avazu.app.libfm.3.prd.no.id > avazu.app.libfm.3.log
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.site.sp.ffm.svm -train kaggle-avazu/base/tr.r0.site.sp.ffm.svm -out kaggle-avazu/base/avazu.site.libfm.3.prd.no.id > avazu.site.libfm.3.log

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/avazu.app.libfm.3.prd.no.id kaggle-avazu/base/avazu.app.libfm.3.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/avazu.site.libfm.3.prd.no.id kaggle-avazu/base/avazu.site.libfm.3.prd
cd kaggle-avazu/base/

./util/pickle_prediction.py avazu.app.libfm.3.prd avazu.app.libfm.3.prd.pickle
./util/pickle_prediction.py avazu.site.libfm.3.prd avazu.site.libfm.3.prd.pickle

./util/merge_prediction.py avazu.app.libfm.3.prd.pickle avazu.site.libfm.3.prd.pickle va.r0.prd.libfm.3.pickle
./util/unpickle_prediction.py va.r0.prd.libfm.3.pickle base.r0.prd.libfm.3

cd ..
cd ..

# libfm model
dimension=1,1,100
iteration=50
method=als
regular=1,1,40

../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.app.sp.ffm.svm -train kaggle-avazu/base/tr.r0.app.sp.ffm.svm -out kaggle-avazu/base/avazu.app.libfm.4.prd.no.id > avazu.app.libfm.4.log
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test kaggle-avazu/base/va.r0.site.sp.ffm.svm -train kaggle-avazu/base/tr.r0.site.sp.ffm.svm -out kaggle-avazu/base/avazu.site.libfm.4.prd.no.id > avazu.site.libfm.4.log

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/avazu.app.libfm.4.prd.no.id kaggle-avazu/base/avazu.app.libfm.4.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/avazu.site.libfm.4.prd.no.id kaggle-avazu/base/avazu.site.libfm.4.prd
cd kaggle-avazu/base/

./util/pickle_prediction.py avazu.app.libfm.4.prd avazu.app.libfm.4.prd.pickle
./util/pickle_prediction.py avazu.site.libfm.4.prd avazu.site.libfm.4.prd.pickle

./util/merge_prediction.py avazu.app.libfm.4.prd.pickle avazu.site.libfm.4.prd.pickle va.r0.prd.libfm.4.pickle
./util/unpickle_prediction.py va.r0.prd.libfm.4.pickle base.r0.prd.libfm.4

cd ..
cd ..

# libilnear-cd
echo -n "liblinear-cd, c = 2"
../../solvers/liblinear-2.1/train -s 7 -c 2 kaggle-avazu/base/tr.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.liblinear.cd.model > avazu.app.liblinear.cd.log
../../solvers/liblinear-2.1/predict -b 1  kaggle-avazu/base/va.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.liblinear.cd.model kaggle-avazu/base/va.r0.app.sp.liblinear.cd.prd.no.id.raw
../../solvers/liblinear-2.1/train -s 7 -c 2 kaggle-avazu/base/tr.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.liblinear.cd.model > avazu.site.liblinear.cd.log
../../solvers/liblinear-2.1/predict -b 1  kaggle-avazu/base/va.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.liblinear.cd.model kaggle-avazu/base/va.r0.site.sp.liblinear.cd.prd.no.id.raw
cat kaggle-avazu/base/va.r0.app.sp.liblinear.cd.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.app.sp.liblinear.cd.prd.no.id
cat kaggle-avazu/base/va.r0.site.sp.liblinear.cd.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.site.sp.liblinear.cd.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.liblinear.cd.prd.no.id kaggle-avazu/base/va.r0.app.sp.liblinear.cd.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.liblinear.cd.prd.no.id kaggle-avazu/base/va.r0.site.sp.liblinear.cd.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.liblinear.cd.prd va.r0.app.sp.liblinear.cd.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.liblinear.cd.prd va.r0.site.sp.liblinear.cd.prd.pickle

./util/merge_prediction.py va.r0.app.sp.liblinear.cd.prd.pickle va.r0.site.sp.liblinear.cd.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.liblinear.cd

cd ..
cd ..

# libilnear-tron
echo -n "liblinear-tron, c = 2"
../../solvers/liblinear-2.1/train -s 0 -c 2 kaggle-avazu/base/tr.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.liblinear.tron.model > avazu.app.liblinear.tron.log
../../solvers/liblinear-2.1/train -s 0 -c 2 kaggle-avazu/base/tr.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.liblinear.tron.model > avazu.site.liblinear.tron.log
../../solvers/liblinear-2.1/predict -b 1  kaggle-avazu/base/va.r0.app.sp.ffm.svm kaggle-avazu/base/tr.r0.app.sp.liblinear.tron.model kaggle-avazu/base/va.r0.app.sp.liblinear.tron.prd.no.id.raw
../../solvers/liblinear-2.1/predict -b 1  kaggle-avazu/base/va.r0.site.sp.ffm.svm kaggle-avazu/base/tr.r0.site.sp.liblinear.tron.model kaggle-avazu/base/va.r0.site.sp.liblinear.tron.prd.no.id.raw
cat kaggle-avazu/base/va.r0.app.sp.liblinear.tron.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.app.sp.liblinear.tron.prd.no.id
cat kaggle-avazu/base/va.r0.site.sp.liblinear.tron.prd.no.id.raw | sed 1d | awk '{print $3}' > kaggle-avazu/base/va.r0.site.sp.liblinear.tron.prd.no.id

./add_id_to_prd.py kaggle-avazu/base/va.r0.app.sp kaggle-avazu/base/va.r0.app.sp.liblinear.tron.prd.no.id kaggle-avazu/base/va.r0.app.sp.liblinear.tron.prd
./add_id_to_prd.py kaggle-avazu/base/va.r0.site.sp kaggle-avazu/base/va.r0.site.sp.liblinear.tron.prd.no.id kaggle-avazu/base/va.r0.site.sp.liblinear.tron.prd

cd kaggle-avazu/base/

./util/pickle_prediction.py va.r0.app.sp.liblinear.tron.prd va.r0.app.sp.liblinear.tron.prd.pickle
./util/pickle_prediction.py va.r0.site.sp.liblinear.tron.prd va.r0.site.sp.liblinear.tron.prd.pickle

./util/merge_prediction.py va.r0.app.sp.liblinear.tron.prd.pickle va.r0.site.sp.liblinear.tron.prd.pickle va.r0.prd.pickle

./util/unpickle_prediction.py va.r0.prd.pickle base.r0.prd.liblinear.tron

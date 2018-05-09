#!/bin/bash
#
git clone https://github.com/guestwalk/kaggle-2014-criteo.git
cd kaggle-2014-criteo
make
cd ..

tar -xzf ../../data/criteo/dac.tar.gz

./kaggle-2014-criteo/converters/txt2csv.py tr train.txt train.csv
./kaggle-2014-criteo/converters/txt2csv.py te test.txt test_without_label.csv
./kaggle-2014-criteo/utils/add_dummy_label.py test_without_label.csv test.csv

ln -sf train.csv tr.csv
ln -sf test.csv te.csv

./kaggle-2014-criteo/utils/count.py tr.csv > fc.trva.t10.txt

./kaggle-2014-criteo/converters/parallelizer-a.py -s 1 ./kaggle-2014-criteo/converters/pre-a.py tr.csv tr.gbdt.dense tr.gbdt.sparse
./kaggle-2014-criteo/converters/parallelizer-a.py -s 1 ./kaggle-2014-criteo/converters/pre-a.py te.csv te.gbdt.dense te.gbdt.sparse

./kaggle-2014-criteo/gbdt -t 0 -s 1 te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse te.gbdt.out tr.gbdt.out

./kaggle-2014-criteo/converters/parallelizer-b.py -s 1 ./kaggle-2014-criteo/converters/pre-b.py tr.csv tr.gbdt.out tr.ffm

./kaggle-2014-criteo/converters/parallelizer-b.py -s 1 ./kaggle-2014-criteo/converters/pre-b.py te.csv te.gbdt.out te.ffm

ln -sf ./tr.ffm ./criteo.tr
ln -sf ./te.ffm ./criteo.va
#
#
#
# linear model
eta=0.2
lambda=0
iteration=13
thread=1

echo -n "Linear, lambda = $lambda, t = $iteration, r = $eta, s = $thread:"
../../solvers/linear/linear-train -l $lambda -t $iteration -r $eta -s $thread  ./criteo.tr criteo.linear.model > criteo.linear.log
../../solvers/linear/linear-predict ./criteo.va criteo.linear.model criteo.linear.output
./script/add_id_to_prediction.py test.csv criteo.linear.output criteo.linear.output.sub

# poly2 model
eta=0.2
lambda=0
iteration=10
thread=1

echo -n "Poly2, lambda = $lambda, t = $iteration, r = $eta, s = $thread:"
../../solvers/poly2/poly2-train -l $lambda -t $iteration -r $eta -s $thread  ./criteo.tr criteo.poly2.model > criteo.poly2.log
../../solvers/poly2/poly2-predict ./criteo.va criteo.poly2.model criteo.poly2.output
./script/add_id_to_prediction.py test.csv criteo.poly2.output criteo.poly2.output.sub

# fm-1 model
eta=0.05
lambda=0.00002
k=40
iteration=8
thread=1

echo -n "FM-1, lambda = $lambda, t = $iteration, r = $eta, s = $thread:"
../../solvers/fm/fm-train -l $lambda -t $iteration -r $eta -s $thread  ./criteo.tr criteo.fm1.model > criteo.fm1.log
../../solvers/fm/fm-predict ./criteo.va criteo.fm1.model criteo.fm1.output
./script/add_id_to_prediction.py test.csv criteo.fm1.output criteo.fm1.output.sub

# fm-2 model
eta=0.05
lambda=0.00002
k=100
iteration=9
thread=1

echo -n "FM-1, lambda = $lambda, t = $iteration, r = $eta, s = $thread:"
../../solvers/fm/fm-train -l $lambda -t $iteration -r $eta -s $thread  ./criteo.tr criteo.fm2.model > criteo.fm2.log
../../solvers/fm/fm-predict criteo.va criteo.fm2.model criteo.fm2.output
./script/add_id_to_prediction.py test.csv criteo.fm2.output criteo.fm2.output.sub

# ffm model
eta=0.2
lambda=0.00002
k=4
iteration=9
thread=1

echo -n "FFM, lambda = $lambda, t = $iteration, r = $eta, s = $thread:"
../../solvers/ffm/ffm-train -l $lambda -t $iteration -r $eta -s $thread  ./criteo.tr criteo.ffm.model > criteo.ffm.log
../../solvers/ffm/ffm-predict ./criteo.va criteo.ffm.model criteo.ffm.output
./script/add_id_to_prediction.py test.csv criteo.ffm.output criteo.ffm.output.sub


python script/ffm2svm.py criteo.tr
python script/ffm2svm.py criteo.va

# poly2-hash model
echo -n "poly2-hash, c = 2"
../../solvers/liblr-poly2-hash/train-p2hash -s 7 -c 2 -h 3 -n 24 ./criteo.tr.svm criteo.poly2.hash.model > criteo.poly2.hash.log
../../solvers/liblr-poly2-hash/predict-p2hash -b 1 ./criteo.va.svm criteo.poly2.hash.model criteo.poly2.hash.output.raw
cat criteo.poly2.hash.output.raw | sed 1d | awk '{print $3}' > criteo.poly2.hash.output
./script/add_id_to_prediction.py test.csv criteo.poly2.hash.output criteo.poly2.hash.output.sub


# libilnear-cd
echo -n "liblinear-cd, c = 2"
../../solvers/liblinear-2.1/train -s 7 -c 2 ./criteo.tr.svm criteo.liblinear.cd.model > criteo.liblinear.cd.log
../../solvers/liblinear-2.1/predict -b 1 ./criteo.va.svm criteo.liblinear.cd.model criteo.liblinear.cd.output.raw
cat criteo.liblinear.cd.log | sed 1d | awk '{print $3}' > criteo.liblinear.cd.output
./script/add_id_to_prediction.py test.csv criteo.liblinear.cd.output criteo.liblinear.cd.output.sub

# libilnear-tron
echo -n "liblinear-tron, c = 2"
../../solvers/liblinear-2.1/train -s 0 -c 2 ./criteo.tr.svm criteo.liblinear.tron.model > criteo.liblinear.tron.log
../../solvers/liblinear-2.1/predict -b 1 ./criteo.va.svm criteo.liblinear.tron.model criteo.liblinear.tron.output.raw
cat criteo.liblinear.tron.log | sed 1d | awk '{print $3}' > criteo.liblinear.tron.output
./script/add_id_to_prediction.py test.csv criteo.liblinear.tron.output criteo.liblinear.tron.output.sub

# libfm model
dimension=1,1,40
iteration=20
method=als
regular=1,1,40

echo -n "LibFM, dimension = $dimension, t = $iteration, method = $method, regular = $regular:"
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test ./criteo.va.svm -train criteo.tr.svm -out criteo.libfm.1 > criteo.libfm.1.log
./script/add_id_to_prediction.py test.csv criteo.libfm.1 criteo.libfm.1.sub

# libfm model
dimension=1,1,40
iteration=50
method=als
regular=1,1,40

echo -n "LibFM, dimension = $dimension, t = $iteration, method = $method, regular = $regular:"
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test ./criteo.va.svm -train criteo.tr.svm -out criteo.libfm.2 > criteo.libfm.2.log
./script/add_id_to_prediction.py test.csv criteo.libfm.2 criteo.libfm.2.sub

# libfm model
dimension=1,1,100
iteration=20
method=als
regular=1,1,40

echo -n "LibFM, dimension = $dimension, t = $iteration, method = $method, regular = $regular:"
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test ./criteo.va.svm -train criteo.tr.svm -out criteo.libfm.3 > criteo.libfm.3.log
./script/add_id_to_prediction.py test.csv criteo.libfm.3 criteo.libfm.3.sub

# libfm model
dimension=1,1,100
iteration=50
method=als
regular=1,1,40

echo -n "LibFM, dimension = $dimension, t = $iteration, method = $method, regular = $regular:"
../../solvers/libfm/bin/libFM -dim $dimension -iter $iteration  -method $method -regular $regular -task c -test ./criteo.va.svm -train criteo.tr.svm -out criteo.libfm.4 > criteo.libfm.4.log
./script/add_id_to_prediction.py test.csv criteo.libfm.4 criteo.libfm.4.sub

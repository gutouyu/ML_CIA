#!/bin/bash


git clone https://github.com/guestwalk/kaggle-2014-criteo.git
cd kaggle-2014-criteo
make
cd ..

tar -xzf ../../data/criteo/dac.tar.gz

./kaggle-2014-criteo/converters/txt2csv.py tr train.txt train.csv
./kaggle-2014-criteo/converters/txt2csv.py te test.txt test_without_label.csv
./kaggle-2014-criteo/utils/add_dummy_label.py test_without_label.csv test.csv


head -n 39800000 ./train.csv > criteo.r1.tr
tail -n 6040618 ./train.csv > criteo.r1.va.no.header
head -n 1 criteo.r1.tr > criteo.r1.va
cat criteo.r1.va.no.header >> criteo.r1.va

./script/sample.py criteo.r1.tr criteo.r10.tr 0.1
./script/sample.py criteo.r1.va criteo.r10.va 0.1

ln -s criteo.r10.tr tr.csv
ln -s criteo.r10.va te.csv

./kaggle-2014-criteo/utils/count.py tr.csv > fc.trva.t10.txt

./kaggle-2014-criteo/converters/parallelizer-a.py -s 4 ./kaggle-2014-criteo/converters/pre-a.py tr.csv tr.gbdt.dense tr.gbdt.sparse
./kaggle-2014-criteo/converters/parallelizer-a.py -s 4 ./kaggle-2014-criteo/converters/pre-a.py te.csv te.gbdt.dense te.gbdt.sparse

./kaggle-2014-criteo/gbdt -t 0 -s 1 te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse te.gbdt.out tr.gbdt.out

rm -f te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse

./kaggle-2014-criteo/converters/parallelizer-b.py -s 4 ./kaggle-2014-criteo/converters/pre-b.py tr.csv tr.gbdt.out tr.ffm

./kaggle-2014-criteo/converters/parallelizer-b.py -s 4 ./kaggle-2014-criteo/converters/pre-b.py te.csv te.gbdt.out te.ffm
rm -f te.gbdt.out tr.gbdt.out


for lambda in 0.001 0.0001 0.00001 0.000001; do 
    echo -n "FFM, lambda = $lambda"
    ../../solvers/ffm/ffm-train -k 4 -l $lambda -r 0.2 -t 150 -s 1 -p te.ffm tr.ffm > ffm.lambda.$lambda
done

for eta in 0.01 0.02 0.05 0.1 0.2 0.5; do 
    echo -n "FFM, eta = $eta"
    ../../solvers/ffm/ffm-train -k 4 -l 0.00002 -r $eta -t 25 -s 1 -p te.ffm tr.ffm > ffm.eta.$eta
done

for thread in 1 2 4 6 8 10 12; do 
    echo -n "FFM, thread = $thread"
    ../../solvers/ffm/ffm-train -k 4 -l 0.00002 -r 0.2 -t 25 -s $thread -p te.ffm tr.ffm > ffm.thread.$thread
done

for k in 1 2 4 6 8 16; do 
    echo -n "FFM, k = $k"
    ../../solvers/ffm/ffm-train -k $k -l 0.00002 -r 0.2 -t 25 -s 1 -p te.ffm tr.ffm > ffm.k.$k
done

python draw_lambda_criteo.py
python draw_eta_criteo.py
python draw_thread_criteo.py
python draw_speedup_criteo.py

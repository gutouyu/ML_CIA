#!/usr/bin/env bash

cd ./libffm_toy/
wget https://www.csie.ntu.edu.tw/~f01922139/libffm_data/libffm_toy.zip
unzip libffm_toy.zip
rm -rf libffm_toy.zip
cd ..
./ffm-train -l 0.0001 -k 15 -t 30 -r 0.05 -s 10 --auto-stop -p libffm_toy/criteo.va.r100.gbdt0.ffm libffm_toy/criteo.tr.r100.gbdt0.ffm model
./ffm-predict ./libffm_toy/criteo.va.r100.gbdt0.ffm model output


echo "kdd12"
./cvt.kddcup2012.py --full kddcup2012.txt kdd12
rm selected.txt

./split.py 0.2 kdd12.ffm kdd12.trva.ffm kdd12.te.ffm
./split.py 0.2 kdd12.trva.ffm kdd12.tr.ffm kdd12.va.ffm

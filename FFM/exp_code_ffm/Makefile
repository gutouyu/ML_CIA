all: solvers/linear-train solvers/poly2_w_linear-train solvers/fm_w_linear-train solvers/ffm_w_linear-train solvers/ffm-train solvers/poly2-train solvers/fm-train solvers/libFM solvers/train-p2hash solvers/liblr-train

solvers/linear-train:
	make -C solvers/linear
	ln -sf linear/linear-train solvers/linear-train 
	ln -sf linear/linear-predict solvers/linear-predict 

solvers/poly2_w_linear-train:
	make -C solvers/poly2_w_linear
	ln -sf poly2_w_linear/poly2_w_linear-train solvers/poly2_w_linear-train 
	ln -sf poly2_w_linear/poly2_w_linear-predict solvers/poly2_w_linear-predict 

solvers/fm_w_linear-train:
	make -C solvers/fm_w_linear/
	ln -sf fm_w_linear/fm_w_linear-train solvers/fm_w_linear-train 
	ln -sf fm_w_linear/fm_w_linear-predict solvers/fm_w_linear-predict 

solvers/ffm_w_linear-train:
	make -C solvers/ffm_w_linear/
	ln -sf ffm_w_linear/ffm_w_linear-train solvers/ffm_w_linear-train
	ln -sf ffm_w_linear/ffm_w_linear-predict solvers/ffm_w_linear-predict

solvers/ffm-train:
	make -C solvers/ffm/
	ln -sf ffm/ffm-train solvers/ffm-train
	ln -sf ffm/ffm-predict solvers/ffm-predict

solvers/poly2-train:
	make -C solvers/poly2/
	ln -sf poly2/poly2-train solvers/poly2-train
	ln -sf poly2/poly2-predict solvers/poly2-predict

solvers/fm-train:
	make -C solvers/fm/
	ln -sf fm/fm-train solvers/fm-train
	ln -sf fm/fm-predict solvers/fm-predict

solvers/libFM:
	make -C solvers/libfm/
	ln -sf libfm/bin/libFM solvers/libFM

solvers/train-p2hash:
	make -C solvers/liblr-poly2-hash/

solvers/liblr-train:
	make -C solvers/liblinear-2.1

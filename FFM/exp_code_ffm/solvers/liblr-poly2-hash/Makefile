CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
SHVER = 3
OS = $(shell uname)
#LIBS = -lblas

all: train-p2hash predict-p2hash

lib: linear.o tron.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train-p2hash: tron.o linear.o timer.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train-p2hash train.c tron.o timer.o linear.o $(LIBS)

predict-p2hash: tron.o linear.o timer.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict-p2hash predict.c tron.o timer.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -std=c++11 -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -std=c++11 -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) -std=c++11 -c -o $@ $<

clean:
	make -C blas clean
	make -C matlab clean
	rm -f *~ tron.o timer.o linear.o train-p2hash predict-p2hash liblinear.so.$(SHVER)

CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

DFLAG = -DUSESSE

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: poly2_w_linear-train

poly2_w_linear-train: ffm-train.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^

poly2_w_linear-predict: ffm-predict.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f ffm-train ffm-predict ffm.o

all:
	eval c++ -O3 -Wall -shared -std=c++11 -fPIC $(shell python3.12 -m pybind11 --includes) src/bilinterp.cpp -o src/bilinterp$(shell python3.12-config --extension-suffix)

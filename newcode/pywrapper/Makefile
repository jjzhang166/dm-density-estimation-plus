
all: cicpy

cicpy:
	 swig -python -c++ -o _cicswigwrapper_module.cc cicswig.i
	 swig -python -c++ -o _gadgetswigwrapper_module.cc GADGETSwig.i
	 python setup.py build_ext --inplace

clean:
	rm -fr *.o *.so
	rm -fr _cicswigwrapper_module.cc
	rm -fr _cicswigwrapper.so
	rm -fr cicpy.py
	rm -fr cicpy.pyc
	rm -fr _gadgetswigwrapper_module.cc
	rm -fr GADGETPy.py
	rm -fr *.pyc
	rm -fr build	

from distutils.core import setup, Extension

extension_mod = Extension("_cicpy", ["_cicswigwrapper_module.cc", "cic.cpp"])

setup(name = "cicpy", ext_modules=[extension_mod])

extension_mod = Extension("_GADGETPy", ["_gadgetswigwrapper_module.cc", "GadgetReader/gadgetreader.cpp", "GadgetReader/read_utils.c"])
setup(name = "GADGETPy", ext_modules=[extension_mod])

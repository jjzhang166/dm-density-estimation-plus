from distutils.core import setup, Extension

extension_mod = Extension("_cicswigwrapper", ["_cicswigwrapper_module.cc", "cic.cpp"])

setup(name = "CIC", ext_modules=[extension_mod])
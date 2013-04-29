#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.util import get_platform
import os, shutil

"""
This variable (add_ext) can be set to False if you don't want the C python extension
to be created (i.e. if you are using Windows and don't have the correct compiler).
"""
add_ext = True


"""
If you don't have both numpy and scipy, you can use the limited PCA module. 
Which only uses Numeric. In that case, set this variable (old_numeric) to True.
"""
old_numeric = False



if old_numeric:
  src_path = os.path.join('src', 'numeric_version')
else:
  src_path = os.path.join('src', 'numpy_version')



# Don't try to build C libraries for Jython:
if get_platform ().startswith ('java') or not add_ext:
    ext = None
else:
    ext = [Extension ('c_nipals', [os.path.join(src_path, 'nipals.c')])]

# Copy correct module to cwd
shutil.copy(os.path.join(src_path, 'pca_module.py'), 'pca_module.py')   



setup (name = 'PCA Module',
       version = '1.1',
       description = 'PCA Module for Python',
       author = 'Henning Risvik',
       author_email = 'risvik@gmail.com',
       url = 'http://folk.uio.no/henninri/pca_module', 
       py_modules = ['pca_module'],
       ext_modules = ext)



# Remove copied module       
os.remove('pca_module.py')
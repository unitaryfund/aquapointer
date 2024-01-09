# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python 
""" Setup file """ 
from setuptools import setup, find_packages 
setup(name='aquapointer', 
      version='0.0.1a', 
      description='aquapointer', 
      author='Unitary Fund', 
      packages = find_packages(include=['aquapointer/', 'aquapointer.*']) 
      )

#!/usr/bin/env python 
""" Setup file """ 
from setuptools import setup, find_packages 
setup(name='aquapointer', 
      version='0.0.1a', 
      description='aquapointer', 
      author='Unitary Fund', 
      packages = find_packages(include=['aquapointer', 'aquapointer.processor']) )


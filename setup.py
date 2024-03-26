# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python
""" Setup file """
from setuptools import setup, find_packages

setup(
    name="aquapointer",
    version=open("VERSION.txt", "r").read().strip(),
    description="aquapointer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Unitary Fund",
    packages=find_packages(include=["aquapointer", "aquapointer.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)

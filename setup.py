# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python
""" Setup file """

from pip._internal import req
from pip._internal.network.session import PipSession

from setuptools import setup, find_packages


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

links = []
requires = []

try:
    requirements = req.parse_requirements('requirements.txt')
except:
    # new versions of pip requires a session
    requirements = req.parse_requirements(
        'requirements.txt', session=PipSession())


for item in requirements:
    # we want to handle package names and also repo urls
    if getattr(item, 'url', None):  # older pip has url
        links.append(str(item.url))
    if getattr(item, 'link', None): # newer pip has link
        links.append(str(item.link))
    try:
        requires.append(str(item.req))
    except:
        requires.append(str(item.requirement))


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
    install_requires=requires,
)

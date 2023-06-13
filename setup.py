#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
   name='modules',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.example',
   packages=find_packages(),
   install_requires=[],
   include_package_data=True
)


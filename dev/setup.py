#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
   name='n-fx-dev',
   version='1.0',
   description='Neural effects development',
   author='Maksim',
   author_email='mkoltugin@gmail.com',
   packages=find_packages(),
   install_requires=["torch"],
   include_package_data=True
)


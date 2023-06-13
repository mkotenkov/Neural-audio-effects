from setuptools import setup

setup(
   name='foo',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.example',
   packages=["modules"],
   install_requires=["torch", "torchaudio", "numpy"],
)
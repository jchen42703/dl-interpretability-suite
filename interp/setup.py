from setuptools import setup

setup(name='interp',
      version='0.00.1',
      description='Library for model interpretability',
      url='',
      author='Joseph Chen, Benson Jin',
      author_email='jchen42703@gmail.com',
      license='Apache License Version 2.0, January 2004',
      packages=['interp'],
      install_requires=[
            'numpy',
            'torch',
      ],
      keywords=['deep learning', 'interpretability'],
      )

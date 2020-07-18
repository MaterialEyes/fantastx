#!/usr/bin/env python

import os
from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='FANTASTX',
        version='0.0',
        description='Fully Automated Nanoscale To Atomic Scale from Theory and\
                        eXperiments',
        packages=find_packages(),
        install_requires=['pymatgen'],
        package_data={},
        author='V. S. Chaitanya Kolluru',
        author_email='kvs.chaitanya@ufl.edu',
        url='https://gitlab.com/MaterialEyes/fantastx/tree/master/Fantastx-19',
        scripts=[os.path.join(module_dir, 'files/run_Jun22.py')]
    )

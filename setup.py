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
        install_requires=['matplotlib', 'numpy', 'scipy',
                          'pymatgen', 'ase>=3.19.0', 'sklearn'],
        package_data={},
        author='V. S. Chaitanya Kolluru, Davis Unruh',
        author_email='vkolluru@anl.gov, dunruh@anl.gov',
        url='https://gitlab.com/MaterialEyes/fantastx/tree/master/Fantastx-19',
        scripts=[os.path.join(module_dir, 'files/run_fx.py')]
    )

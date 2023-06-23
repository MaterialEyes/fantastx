#!/usr/bin/env python

import os
from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='FANTASTX',
        version='0.0',
        description='Fully Automated Nanoscale To Atomic Scale from Theory and eXperiments',
        long_description='Fully Automated Nanoscale To Atomic Scale from Theory and eXperiments',
        packages=find_packages(),
        install_requires=[
            'numpy>=1.23,<1.25',
            'matplotlib>=3.7.1',
            'scipy>=1.10.1',
            'pandas>=2.0.2',
            'pymongo>=4.3.3',
            'scikit-learn>=1.2.2',
            'dscribe>=2.0.0',
            'dask[complete]',
            'dask-jobqueue>=0.8.1',
            'PyOpenGL>=3.1.7',
            'PyOpenGL_accelerate>=3.1.7',
            'Pillow>=9.5.0',
            'h5py>=3.8.0',
            'imageio>=2.31.0',
            'requests>=2.31.0',
            'scikit-image>=0.21.0',
            'ase>=3.22.1',
            'pymatgen>=2023.5.31',
            'opencv-python>=4.7.0.72'
        ],
        package_data={},
        author='V. S. Chaitanya Kolluru, Davis Unruh',
        author_email='vkolluru@anl.gov, dunruh@anl.gov',
        url='https://github.com/MaterialEyes/fantastx',
        scripts=[os.path.join(module_dir, 'files/run_fx.py'),
                 os.path.join(module_dir, 'files/run_fx_mp.py')],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3 :: Only"
        ],
        python_requires=">=3.7"
    )

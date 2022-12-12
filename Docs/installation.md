# Installing base FANTASTX

***

## Setting up a virtual environment

It is recommended to start installation of FANTASTX on a new virtual environment using Anaconda.

### Setting up Anaconda

Load Anaconda if available as a library
```sh
module load conda
```
Or in some systems, Anaconda comes with python module. So
```sh
module load python
```

If above methods fail, download Anaconda for the system [here](https://docs.conda.io/en/latest/miniconda.html). Follow default instructions and install Anaconda. Restart the terminal so that the installation takes effect.

Update conda and add conda-forge to the Anaconda channels & set higher priority. Skip this step if conda-forge is already added before. Install everything from conda-forge to be consistent & reduce compatibility issues across different pacakges.
```sh
conda update -n base -c defaults conda
conda config --add channels conda-forge
```

### Creating the conda environment

Create a new conda environment to house FANTASTX in, and set the python version of the new environment to be 3.x. We recommend the use of the most recent version of python3 available (currently python 3.11 at the time of writing), unless you will be performing PDF simulations, in which case python 3.7 must be used.

This installation can be done in two ways. Manually creating and setting the conda path:

```sh
conda create -yp ~/miniconda3/envs/fantastx
conda activate ~/miniconda3/envs/fantastx
export PATH=~/miniconda3/envs/fantastx/bin:$PATH
conda install python=3.11
```

or automatically creating it:

```sh
conda create -n fantastx python=3.11
```

Both of these methods will result in a new environment called *fantastx* being created, which can be activated by running:

```sh
conda activate fantastx
```

## Installing FANTASTX

Install Fantastx by cloning this repository. Enter username and password when prompted. Install via the setup.py using the keyword *develop* for ease of updating the code during the development phase.

```sh
git clone https://github.com/MaterialEyes/fantastx.git

python setup.py develop
```

This will automatically download all of the required FANTASTX packages, including *pymatgen*, *dscribe*, *sklearn* and *ase*. 

# Optional dependencies

***

FANTASTX has several optional features that can accessed by installing the requisite packages. These are not only limited to various forward simulation methods, but also parallelization schemes.

## Parallelization

FANTASTX can be run in parallel using either multiprocessing, or Dask. We recommend the use of multiprocessing on high performance computing systems with long queue times, and Dask on high performance computing systems where queues are shorter and multiple nodes open for use within the span of hours. If utilizing Dask, you must install the Dask and Dask-jobqueue packages:

```sh
conda install -c conda-forge dask dask-jobqueue
```

## TEM and STM Support

Follow instructions [here](https://github.com/MaterialEyes/ingrained/blob/dev_ch/README.md) to install Ingrained package (for TEM and STM simulations). 

## XAS Support

Follow instructions [here](https://github.com/MaterialEyes/xtk) to install the Xanes ToolKit (xtk) package for XANES and XTA simulations.

## PDF Support

PDF simulations are performed using the Diffpy package. This package can be installed via:
```sh
conda install -c diffpy diffpy-cmi
```

Note that the installation of diffpy requires python=3.7 or lower. If installing diffpy, set the python version of the virtual environment you are installing into to be 3.7.

## XRD Support

XRD simulation are performed using GSASII. This package can be installed via:

```sh
conda install gsas2pkg -c defaults -c conda-forge -c briantoby 
```

See [this](https://subversion.xray.aps.anl.gov/trac/pyGSAS) page for more information on installation.

## Documentation

If you want to contribute to the documentation, then you will need to install mkdocs and a few mkdocs extensions. The base mkdocs and its extension can be installed via:

```sh
pip install mkdocs-material
pip install mkdocs-jupyter
pip install mkdocstrings[python]
```

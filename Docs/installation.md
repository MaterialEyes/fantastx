# Installing base FANTASTX

***

## Setting up a virtual environment

It is recommended to start installation of FANTASTX on a new virtual environment using Anaconda. This is not strictly necessary, but is recommended in order to reduce incompatibility issues.

### Setting up Anaconda

If you are setting up FANTASTX on a cluster, then first try to load Anaconda as a module or library. If Anaconda is available as a library, then the command will be similar to:
```sh
module load conda
```
Alternatively, in some systems, Anaconda comes with the Python module. In these cases, the command will be similar to:
```sh
module load python
```
These exact commands will vary depending on the cluster and available modules/ libraries. 

If above methods fail, or you are installing FANTASTX on your local machine and need to install Anaconda, you can download Anaconda for the system [here](https://docs.conda.io/en/latest/miniconda.html). Follow the default instructions and install Anaconda. Restart the terminal so that the installation takes effect. Make sure you update Anaconda to the most recent version by running:
```sh
conda update -n base -c defaults conda
```

### Creating the conda environment

Now that Anaconda has either been activated or installed, you can create the conda environment in which you will install FANTASTX, and set the python version to be 3.x. The python version recommended depends on the types of simulations that will be performed. A table of the python version required for the different simulation architectures is below, current as of 6/7/2023. For the broadest coverage, we recommend the use of python 3.10.

| Forward Simulation | Code     | Most Recent Python Version Supported |
| -------------------- | ----------------------------- | ------------- |
| `TEM or STM`         | Ingrained (incostem and VASP) | 3.11          |
| `PDF`                | DiffPy                        | 3.10 (*3.7)   |
| `XRD`                | GSASII                        | 3.10          |
| `XAS`                | XTK                           | 3.11          |

*If you are installing DiffPy via conda and diffpy-cmi, then 3.7 is the most recent Python version supported.

Proceeding with installation can be done in two ways. Manually creating and setting the conda path:

```sh
conda create -yp ~/miniconda3/envs/fantastx
conda activate ~/miniconda3/envs/fantastx
export PATH=~/miniconda3/envs/fantastx/bin:$PATH
conda install python=3.10
```

or automatically creating it:

```sh
conda create -n fantastx python=3.10
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

The forward simulation capabilities of FANTASTX rely on the support of external packages. These packages must be installed independently in order to utilize them with FANTASTX. For all package installations, ensure that installation occurs within the same conda environment that FANTASTX was installed in.

## TEM and STM support

Follow instructions [here](https://github.com/MaterialEyes/ingrained/blob/dev_ch/README.md) to install Ingrained package (for TEM and STM simulations). Ingrained uses incostem for TEM simulations, which is an open source package. However, Ingrained uses VASP to run STM simulations, so the user will need a VASP license to run STM simulations. If the use of an open source DFT code such as Quantum Espresso is necessary, contact the FANTASTX team and we can see what we can do.

## XAS Support

Follow instructions [here](https://github.com/MaterialEyes/xtk) to install the Xanes ToolKit (xtk) package for XANES and XTA simulations. This package manages setting up, running, and analyzing all the XANES and XTA simulations.

However, the xtk package does not run the XANES and XTA simulations itself. For that, you must also download the [FEFF](https://github.com/times-software/feff10/releases/tag/v10.0.0) or [FDMNES](https://fdmnes.neel.cnrs.fr/download/) packages.

To download and install FEFF, follow the following protocol (once you have registered for FEFF10):

```sh
wget https://github.com/times-software/feff10/archive/refs/tags/v10.0.0.tar.gz
tar -xvf v10.0.0.tar.gz
mv feff10-10.0.0 feff10
cd feff10
cd src
cp Compiler.mk.default Compiler.mk
make mpi
make clean
```

This protocol will attempt to use the compilers ifort (for FORTRAN90), mpif90 (for MPI), and no MKL. If you want to switch the compilation options or turn on MKL support, then edit the Compiler.mk.default (and subsequently the Compiler.mk) file in the src directory. 

To download and install FDMNES, you just need to download the `parallel_fdmnes` package from the [FDMNES download](https://fdmnes.neel.cnrs.fr/download/) page. This package ships with a pre-compiled version of FDMNES which works on both clusters and local linux capable machines. If other support is necessary, then follow the FDMNES instructions for other packages. If issues are encountered, contact the FANTASTX team.

## PDF Support

PDF simulations are performed using the [DiffPy](https://www.diffpy.org/) package. There are two ways to install this package. It can be installed seamlessly with conda via:
```sh
conda install -c diffpy diffpy-cmi
```

If you install DiffPy in this manner, all of the packages developed by the DiffPy team (including those required by FANTASTX!) will be installed. However, not all DiffPy packages are actively being developed, and thus the most recent version of Python which is supported is Python 3.7.

Alternatively, this restriction can be bypassed by utilizing a combination of conda and pip installation. Making sure that you have the conda environment that FANTASTX was installed in activated, run the commands:

```sh
conda install -c conda-forge pyobjcryst
conda install -c conda-forge libdiffpy
conda install -c conda-forge diffpy.structure
pip install 'diffpy.srreal @ git+https://github.com/diffpy/diffpy.srreal'
pip install 'diffpy.srfit @ git+https://github.com/diffpy/diffpy.srfit'
```

This method currently (as of 6/2023) works for python 3.10 but not python 3.11.

## XRD Support

XRD simulation are performed using GSASII. We recommend using one of two simple installation methods.

1. Using anaconda, the package can be installed via:

    ```sh
    conda install gsas2pkg -c defaults -c conda-forge -c briantoby 
    ```

    This will install the entire GSASII package ready for use. However, as of writing (6/2023), this method fails due to wxPython and subversion version issues in the native anaconda channel, and has not been re-worked to utilize conda-forge internally for this problem. If it fails, use the other method.

2. If the above method doesn't work, we can download GSASII and install GSASIIscriptable (which FANTASTX needs) via the following commands:
    ```sh
    conda install -c conda-forge subversion
    conda install -c conda-forge wxPython=4.2
    svn co https://subversion.xray.aps.anl.gov/pyGSAS/trunk /loc/GSASII
    python /loc/GSASII/GSASIIscriptable.py
    ```
    In this method, replace `/loc` with the directory that you will be putting the GSASII folder in. It can also be noted that you can start the GSASII GUI by running
    ```sh
    python /loc/GSASII/GSASII.py
    ```
    which is extremely useful for fitting the initial XRD parameters.

See [this](https://subversion.xray.aps.anl.gov/trac/pyGSAS) page and [this](https://gsas-ii.readthedocs.io/en/latest/GSASIIscriptable.html#commandlineinterface) page for more information on installation.

## Documentation

If you want to contribute to the documentation, then you will need to install mkdocs and a few mkdocs extensions. The base mkdocs and the required extensions can be installed via:

```sh
pip install mkdocs-material
pip install mkdocs-jupyter
pip install mkdocstrings[python]
pip install mkdocs-git-revision-date-plugin
pip install jupyter_contrib_nbextensions
```

## HPC Installation

FANTASTX installation has been tested on Argonne's Carbon and LCRC systems, as well as NERSC. Installation notes are as follows.

### LCRC

Installation has been tested for the python=3.10 procedure above. To load the necessary modules and Anaconda, run the following commands:

```sh
module load gcc mpi mkl anaconda3/2021.05
eval "$(conda shell.bash hook)"
```

When installing GSASII, you might find that you need to also run the command:

```sh
export LC_CTYPE=en_US.UTF-8
```

### Carbon

Installation has been tested for the python=3.10 procedure above. To load the necessary modules and Anaconda, run the following commands:

```sh
module load intel impi gcc
source /opt/apps/anaconda3/4.0.0-2-EL6/etc/profile.d/conda.sh
```

When installing GSASII, you might find that you need to also run the command:

```sh
export LC_CTYPE=en_US.UTF-8
```

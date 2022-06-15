## FANTASTX - Fully Automated from Nanoscale To Atomic Structure from Theory And eXperiments

FANTASTX finds structures of grain boundaries or clusters by performing genetic algorithm with multi objective optimization. The energy of the structure shall be lowered as well as how good a structure match with experimental data like TEM image or PDF data or XPS data.

## Installation

It is recommended to start installation on a new conda environment using Anaconda.

Load Anaconda if available as a library
```sh
module load conda
```
Or in some systems, Anaconda comes with python module. So
```sh
module load python
```

If above methods fail, download Anaconda for the system [here](https://docs.conda.io/en/latest/miniconda.html). Follow default instructions and install Anaconda. Restart the terminal so that the installation takes effect.

Update conda and add conda-forge to the Anaconda channels & set higher priority. This step can be skipped if conda-forge has been previously added, or if adding the conda-forge flags directly to each conda installation step (adding the flags --channel conda-forge). Installing everything from conda-forge is desired to maintain consistency & reduce compatibility issues across different packages.

```sh
conda update -n base -c defaults conda
conda config --add channels conda-forge
```

Create a new conda environment with path. Provide *path* to the new conda environment. Add the new environment (*fantastx*) to the system path. Install python 3. If compatability issues are encountered, explicitly set python=3.7.

```sh
conda create -yp ~/miniconda3/envs/fantastx
conda activate ~/miniconda3/envs/fantastx
export PATH=~/miniconda3/envs/fantastx/bin:$PATH
conda install python=3
```

Install following dependencies in this order -

1. Diffpy (For PDF simulation)
  ```sh
  conda install -c diffpy diffpy-cmi
  ```

2. Install Pymatgen. Prerequisites to installing pymatgen are numpy, scipy and matplotlib. If ASE functionality is desired, then also install ASE prior to installing pymatgen by running:
  ```sh
  pip install --upgrade --user ase
  ```
  Follow this by installing pymatgen using either pip: 
  ```sh
  conda install --yes numpy scipy matplotlib
  pip install pymatgen
  ```
  or using conda:
  ```sh
  conda install pymatgen
  ```

 Note that the conda installation of pymatgen will automatically install the numpy, scipy and matplotlib dependencies. As ASE is an optional dependency, it must be explicitly installed prior to installing pymatgen. 

3. Dask and Dask-jobqueue (for parallel calculations on local, SLURM or PBS clusters)
 ```sh
 conda install -c conda-forge dask dask-jobqueue
 ```

4. If using fingerprints, install the sci-kit learn package:
 ```sh
 conda install scikit-learn
 ```
 and if using any dscribe fingerprints then install dscribe:
 ```sh
 conda install dscribe
 ```
 Also install the sci-kit learn package

5. If performing TEM or STM simulations, follow instructions [here](https://github.com/MaterialEyes/ingrained/blob/dev_ch/README.md) to install the Ingrained package.

Finally, install Fantastx by cloning this repository. Enter username and password when prompted. Install using *develop* for ease of updating the code during development phase.

```sh
git clone https://github.com/MaterialEyes/fantastx.git

python setup.py develop
```

## Usage
Find the usage of Fantastx [here] (https://github.com/MaterialEyes/fantastx/blob/master/USAGE.md)

## Citation
If you find this code useful, please consider citing our [paper](#paper)
```sh
@article{,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

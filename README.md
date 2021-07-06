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

Update conda and add conda-forge to the Anaconda channels & set higher priority. Skip this step if conda-forge is already added before. Install everything from conda-forge to be consistent & reduce compatibility issues across different pacakges.
```sh
conda update -n base -c defaults conda
conda config --add channels conda-forge
```

Create a new conda environment with path. Provide path to the new conda environment. Add the new environment to the system path. Install python 3.

```sh
conda create -yp /home/xxxxx/miniconda3/envs/fantastx
conda activate /home/xxxxx/miniconda3/envs/fantastx
export PATH=/home/xxxxx/miniconda3/envs/fx_521/bin:$PATH
conda install python=3
```

Install following dependencies in this order -

Diffpy (For PDF simulation - Optional)
```sh
conda install -c diffpy diffpy-cmi
```

Install Pymatgen (Installs Numpy, Scipy, Matplotlib) with pip instead of conda.

```sh
pip install pymatgen
```

Dask, Dask-jobqueue (for parallel calculations on SLURM/PBS cluster)

```sh
conda install -c conda-forge dask dask-jobqueue
```

Ingrained (For TEM simulation)
Follow instructions [here](https://github.com/MaterialEyes/ingrained/blob/master/README.md)

Install Fantastx by cloning this repository. Enter username and password when prompted. Install using 'develop' for ease of updating the code during development phase.

```sh
git clone https://github.com/MaterialEyes/fantastx.git

python setup.py develop
```

## Usage

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

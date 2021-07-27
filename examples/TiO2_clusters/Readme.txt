-----------------------------------------------------------------
### Get Ingrained
-----------------------------------------------------------------
# get latest ingrained from github
git clone https://github.com/MaterialEyes/ingrained.git
<username & pwd>

# Change branch to dev_ch
git checkout dev_ch

# replace xxxxx with YOUR materialsproject API key
./utils/set_key.sh xxxxxxxxx

# install pydm3 & sxm reader
./utils/install_parsers.sh

# install ase
pip install --upgrade --user ase

# Add path to sys
export PATH=/home/xxxxxxx/.local/bin:$PATH

-----------------------------------------------------------------
### install Ingrained
-----------------------------------------------------------------

# install
python setup.py develop

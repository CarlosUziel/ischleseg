#!/bin/bash

conda create -n diss python=2.7 anaconda -y
echo ". /home/uziel/anaconda2/etc/profile.d/conda.sh" >> ~/.bashrc
source activate diss
conda install matplotlib -y
conda install jupyterlab -y
conda install git -y

pip install argparse
pip install msgpack
pip install theano
cd deepmedic
pip install .
pip install nilearn
pip install nipype
pip install scikit-learn
pip install scikit-image
pip install scipy

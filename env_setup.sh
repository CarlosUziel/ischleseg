#!/bin/bash

conda create -n diss_tf python=2.7 anaconda -y
echo ". /home/uziel/anaconda2/etc/profile.d/conda.sh" >> ~/.bashrc
source activate diss_tf #conda activate diss_tf
conda install matplotlib -y
conda install jupyterlab -y
conda install git -y

pip install argparse
pip install msgpack
pip install --upgrade tensorflow #pip install --upgrade tensorflow-gpu
cd deepmedic
pip install .
pip install nilearn
pip install nipype
pip install scikit-learn
pip install scikit-image
pip install scipy

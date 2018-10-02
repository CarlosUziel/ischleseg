# M.Sc. Dissertation: Ischemic Stroke Lesion Segmentation using Convolutional Neural Networks
(Publication pending)
## Abstract
Ischemic stroke is the most common cerebrovascular disease worldwide and only in the US represents 87% of all strokes. It is produced when a brain artery is obstructed, interrupting the blood flow and eventually killing brain cells. The efficacy of possible treatments highly depends on the time since the stroke onset. Accurately segmenting the stroke lesion (i.e. the affected area of the brain) is a very tedious task that requires examining multiple MRI sequences and has low inter-observer agreement. The stroke lesion can happen anywhere in the brain with different sizes and its characteristics are highly variable. Manual segmentation is the gold standard, but it is impractical in the medical routine. Therefore, it is necessary to develop segmentation algorithms that are able to locate and segment stroke lesions automatically. Such an algorithm is DeepMedic, winner of the first iteration of the Ischemic Stroke Lesion Segmentation Challenge (ISLES). This project uses DeepMedic as core algorithm, but focuses on data pre-processing and post-processing using the ISLES 2017 data set in order to find new ways of improving performance. Of all techniques implemented, data augmentation with binary closing achieved the best results, obtaining an improvement of DICE score of 17% over the baseline model. Also, the results show that DeepMedic performs better for big lesions than for small ones.

### Keywords
Ischemic Stroke, Semantic Segmentation, CNN, data pre-processing, data augmentation, DeepMedic, ISLES.

## Environment setup
To run the contents of this repository, it is recommended to set-up a python virtual environment (preferably with Anaconda), following the script `env_setup.sh`, in the root folder:

```
conda create -n "env_name" python=2.7 anaconda -y
echo ". /home/"user"/anaconda2/etc/profile.d/conda.sh" >> ~/.bashrc
source activate "env_name"
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
```

Beware of replacing `"user"` with the corresponding user name and `"env_name"` with a  suitable environment name.

## Content of this repository
This repository contains all the code used in the master dissertation project. On one hand, it contains the source code of [DeepMedic](https://github.com/Kamnitsask/deepmedic) v0.6.1. The directory tree has been slightly changed to accomodate to the different experiments carried out and their corresponding configuration files (plese referer to the original documentation of [DeepMedic](https://github.com/Kamnitsask/deepmedic) to understand how DeepMedic processes these files). Concretly, each model that was trained can be found under the directory `deepmedic/versions`. The structure of each model is as follows:
- `"model"/configFiles/model`: Model configuration file.
- `"model"/configFiles/train`: Train configuration file and image sequence files.
- `"model"/configFiles/validation`: Validation configuration and image sequence files.
- `"model"/configFiles/test`: Train configuration file and image sequence files.
- `"model"/run_train.sh`: Trains the model.
- `"model"/run_test.sh`: Tests the model.

On the other hand, the main contribution of this project is concerned with data pre-processing and augmentation. The code for this was developed using [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/#), reason why all the code was written in Jupyter notebooks. The main notebooks are as follows:
- `gen_preproceesed_data.ipynb`: Pre-processes data and creates the baseline models' configuration files.
- `gen_augmented_data.ipynb`: Augments data and creates the corresponding models' configuration files. 
- `postprocessing.ipyb`: Performs multiple post-processing steps on the network output to improve results. Also produces all plots for results analysis.

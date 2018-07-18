
# coding: utf-8

# In[ ]:


#%% Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from glob import glob
from scipy import ndimage
from nilearn.image import resample_to_img, resample_img
from nilearn.masking import compute_background_mask, compute_epi_mask
from nilearn.plotting import plot_roi, plot_epi
from scipy.spatial.distance import directed_hausdorff
from nipype.algorithms.metrics import Distance
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from scipy import interp
from itertools import chain
from scipy.ndimage.morphology import binary_dilation, binary_erosion


# In[ ]:


# Set working directory
os.chdir('/home/uziel/DISS')

# Set root of models to be post-processed
root = "./milestones_4"

# Set path for error maps
error_map_path = './data_processed/error_maps'

# Load best performing model variant
model_variant = 'DM_V0_[0-4]' # this must be the best performing model
tmp = model_variant.split('_')
if len(tmp) == 3:
    model_name = tmp[1]
elif len(tmp) == 4:
    model_name = tmp[1] + '_' + tmp[2]
else:
    model_name = tmp[1] + '_' + tmp[2] + '_' + tmp[3]

# Load all trained models (k-folds) of model_variant
trained_models = sorted(glob(os.path.join(root, model_variant)))


# **GET ERROR MAPS**
# 
# Compute error maps for the best performing model. Get probability maps for class 0 and 1 for all validation subjects across all folds (should sum up to 43). Obtain error maps by substracting each subject's probability map from the ground truth. Save error maps. For each trained model, create files listing error maps corresponding to the training subjects (and validation?).

# In[ ]:


# Create folder in data for error maps
if os.path.exists(error_map_path): shutil.rmtree(error_map_path)
os.mkdir(error_map_path)

# Get all probability maps for class 0 and 1 from model variant X (43)
prob_maps_class_0 = []
prob_maps_class_1 = []
for model in trained_models:
    root_1 = os.path.join(model, 'output/predictions/valSession/predictions')

    # Load probability maps of background
    prob_maps_class_0 += (glob(os.path.join(root_1, '*ProbMapClass0.nii.gz')))
    # Load probability maps of foreground
    prob_maps_class_1 += (glob(os.path.join(root_1, '*ProbMapClass1.nii.gz')))

prob_maps_class_0 = sorted(prob_maps_class_0)
prob_maps_class_1 = sorted(prob_maps_class_1)

# Get all ground truth labels for all training subjects (43)
root_2 = './data_processed/ISLES2017/training'
subject_labels = sorted([y
                         for x in os.walk(root_2)
                         for y in glob(os.path.join(x[0], '*OT*.nii.gz'))
                         if 'clone' not in y
                        ])

# Subject code in prediction files comes from MTT channel
subject_mtt = sorted([y
                      for x in os.walk(root_2)
                      for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))
                      if 'clone' not in y
                     ])

# Compute error maps
for i in range(len(subject_mtt)):
    # Load label
    label = nib.load(subject_labels[i])
    
    # Get subject code
    code = os.path.basename(subject_mtt[i]).split('.')[-3]
    
    # Get probability maps of subject code
    pmap_0 = [m for m in prob_maps_class_0 if code in m][0]
    pmap_1 = [m for m in prob_maps_class_1 if code in m][0]
    
    pmap_0_img = nib.load(pmap_0)
    pmap_1_img = nib.load(pmap_1)
    
    # Compute square error map
    emap_0 = ((label.get_data() == 0).astype(int) - pmap_0_img.get_data())**2
    emap_1 = (label.get_data() - pmap_1_img.get_data())**2
    
    # Normalize
    emap_0 = (emap_0 - np.mean(emap_0))/np.std(emap_0)
    emap_1 = (emap_1 - np.mean(emap_1))/np.std(emap_1)
    
    # Save error maps
    nib.save(nib.Nifti1Image(emap_0, pmap_0_img.affine),
             os.path.join(error_map_path, 'EMAP.0.' + code + '.nii.gz'))
    nib.save(nib.Nifti1Image(emap_1, pmap_1_img.affine),
             os.path.join(error_map_path, 'EMAP.1.' + code + '.nii.gz'))
    


# **CREATE CONFIGURATION FILES FOR WEIGHTED MAPS**
# 
# weightedMapsForSamplingEachCategoryTrain = ["./weightMapsForeground.cfg", "./weightMapsBackground.cfg"]
# #weightedMapsForSamplingEachCategoryVal = ["./validation/weightMapsForeground.cfg", "./validation/weightMapsBackground.cfg"]
# 

# In[ ]:


# Load error map paths
emaps_0 = sorted([x for x in os.listdir(error_map_path) if 'EMAP.0.' in x])
emaps_1 = sorted([x for x in os.listdir(error_map_path) if 'EMAP.1.' in x])


# In[ ]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[ ]:


for model in trained_models:
    # Set new model path
    old_model_path = './ischleseg/deepmedic/versions/' + os.path.basename(model)
    new_model_path = './ischleseg/deepmedic/versions/DM_V3_' + model.split('_')[-1]
    if not os.path.exists(new_model_path): shutil.copytree(old_model_path, new_model_path)
    
    # Load train and validation MTT channels
    trainChannels_MTT = sorted([line.rstrip('\n') for line in open(os.path.join(model, 'configFiles/train/trainChannels_MTT.cfg'))])
    validationChannels_MTT = sorted([line.rstrip('\n') for line in open(os.path.join(model, 'configFiles/validation/validationChannels_MTT.cfg'))])
    
    # Get subject codes for train and validation
    train_codes = [x.split('.')[-3] for x in trainChannels_MTT]
    val_codes = [x.split('.')[-3] for x in validationChannels_MTT]
    
    # Get train weight maps
    train_weightMapsBackground = []
    train_weightMapsForeground = []
    for t_code in train_codes:
        train_weightMapsBackground.append([os.path.join('../../../../../../', x)
                                           for x in emaps_1 if x.split('.')[-3] in t_code][0])
        train_weightMapsForeground.append([os.path.join('../../../../../../', x)
                                           for x in emaps_0 if x.split('.')[-3] in t_code][0])
    
    # Save train weight maps
    data_to_file(train_weightMapsBackground,
                 os.path.join(new_model_path, 'configFiles/train/weightMapsBackground.cfg'))
    data_to_file(train_weightMapsForeground,
                os.path.join(new_model_path, 'configFiles/train/weightMapsForeground.cfg'))
    
    
    # Get val weight maps    
    val_weightMapsBackground = []
    val_weightMapsForeground = []
    for t_code in train_codes:        
        val_weightMapsBackground.append([os.path.join('../../../../../../', x)
                                         for x in emaps_1 if x.split('.')[-3] in t_code][0])
        val_weightMapsForeground.append([os.path.join('../../../../../../', x)
                                         for x in emaps_0 if x.split('.')[-3] in t_code][0])

    # Save train weight maps
    data_to_file(val_weightMapsBackground,
                os.path.join(new_model_path, 'configFiles/validation/weightMapsBackground.cfg'))
    data_to_file(val_weightMapsForeground,
                os.path.join(new_model_path, 'configFiles/validation/weightMapsForeground.cfg'))


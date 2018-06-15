
# coding: utf-8

# In[1]:


#%% Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
from random import shuffle
from glob import glob
from scipy import ndimage
from nilearn.image import resample_to_img, resample_img
from nilearn.masking import compute_background_mask, compute_epi_mask
from nilearn.plotting import plot_roi, plot_epi


# In[2]:


#%% Set current directory
os.chdir('/home/he/carlos/DISS')
test_flag = 0


# In[3]:


#%% List all sequences per subject
if test_flag:
    root = './data/ISLES2017/testing'
else:
    root = './data/ISLES2017/training'

subjects_paths = sorted(os.listdir(root))
channels_per_subject = dict() # groups relevant sequences per subject
for i in range(len(subjects_paths)):
    s_path = os.path.join(root, subjects_paths[i])
    channels_per_subject[i] = [y
                               for x in os.walk(s_path)
                               for y in
                               glob(os.path.join(x[0], '*ADC*.nii')) or
                               glob(os.path.join(x[0], '*MTT*.nii')) or
                               glob(os.path.join(x[0], '*rCBF*.nii')) or
                               glob(os.path.join(x[0], '*rCBV*.nii')) or
                               glob(os.path.join(x[0], '*Tmax*.nii')) or
                               glob(os.path.join(x[0], '*TTP*.nii')) or
                               glob(os.path.join(x[0], '*OT*.nii'))
                              ]


# In[4]:


#%% Resample images to same shape and voxel size
# linux
if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'

template_path = './data/MNI152_T1_1mm_brain.nii.gz'

# remove and create dir for processed data
if os.path.exists(root): shutil.rmtree(root)
os.makedirs(root)

# load template image
template = nib.load(template_path)

for subject in channels_per_subject.keys():
    # create subdirectory per subject
    subject_root = os.path.join(root, str(subject))
    os.mkdir(subject_root)
    
    subject_imgs = []
    for channel_file in channels_per_subject[subject]:
        img = nib.load(channel_file)

        # Resample img to match template
        if 'OT' in channel_file:
            # label must be resampled using nearest neighbour
            resampled_img = resample_to_img(img, template, interpolation='nearest')
            # resample image to two thirds of its size (allows for faster and less memory need)
            resampled_img = resample_img(resampled_img,
                                         3*resampled_img.affine/2, [2*x/3 for x in resampled_img.shape],
                                         interpolation='nearest')
        else:
            resampled_img = resample_to_img(img, template, interpolation='continuous')
            # resample image to two thirds of its size (allows for faster and less memory need)
            resampled_img = resample_img(resampled_img,
                             3*resampled_img.affine/2, [2*x/3 for x in resampled_img.shape],
                             interpolation='continuous')

        subject_imgs.append([resampled_img, channel_file])
        
    # compute subject brain mask given all channels (ignore label channel)
    mask = compute_epi_mask([x for x,y in subject_imgs if not 'OT' in y])
    # dilate mask to "fill holes"
    dilated_mask_data = ndimage.binary_dilation(mask.dataobj)
    mask = nib.nifti1.Nifti1Image(dilated_mask_data.astype(np.int32), mask.affine)
    # save mask
    nib.save(mask, os.path.join(subject_root, 'mask.nii.gz'))
    
    # normalize each image within mask and save
    for img, channel_file in subject_imgs:
        # don't try to normalize label channel
        if 'OT' not in channel_file:
            # get data within mask
            temp_data = img.dataobj * mask.dataobj
            # compute mean and variance of non-zero values
            mean = np.mean(temp_data[np.nonzero(temp_data)])
            var = np.var(temp_data[np.nonzero(temp_data)])
            # substract mean and divide by variance all non-zero valuess
            temp_data[np.nonzero(temp_data)] = (temp_data[np.nonzero(temp_data)] - mean) / var
            # build normalised image with normalised data and unmodified affine
            img = nib.nifti1.Nifti1Image(temp_data.astype(np.float32), img.affine)
        else:
            img = nib.nifti1.Nifti1Image(img.get_data().astype(np.int32), img.affine)
        
        # save image
        file_name = os.path.basename(channel_file)
        nib.save(img, os.path.join(subject_root, file_name) + '.gz')
        
    print("Subject " + str(subject) + " finished.")


# In[4]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[32]:


# Generate files listing all images per channel
if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'
    
channels = {}
# channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
channels['Channels_ADC'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))
                                   if 'clone' not in y
                                  ])
channels['Channels_MTT'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))
                                   if 'clone' not in y
                                  ])
channels['Channels_rCBF'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_rCBV'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_Tmax'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_TTP'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))
                                   if 'clone' not in y
                                  ])
# labels
channels['GtLabels'] = sorted([os.path.join('../../../../../../', y)
                               for x in os.walk(root)
                               for y in glob(os.path.join(x[0], '*OT*.nii.gz'))
                               if 'clone' not in y
                              ])
# masks
channels['RoiMasks'] = sorted([os.path.join('../../../../../../', y)
                               for x in os.walk(root)
                               for y in glob(os.path.join(x[0], 'mask.nii.gz'))
                               if 'clone' not in y
                              ])


# In[30]:


######################################
##### FILES FOR DM_V1 (BASELINE) #####
######################################

# number of train and validation sets
n_set = 10

if test_flag:
    # set paths for storing channel config files
    test_path = './ischleseg/deepmedic/versions/DM_V1/configFiles/test'
    if not os.path.exists(test_path): os.makedirs(test_path)
    for name, files in channels.iteritems():
        # save test channel files
        data_to_file(files, os.path.join(test_path, 'test' + name + '.cfg'))
        
    # save names of predictions
    names = ['pred_ISLES2017_' + os.path.basename(x).split('.')[-3] for x in channels['Channels_ADC']]
    data_to_file(names, os.path.join(test_path, 'testNamesOfPredictions.cfg'))
else:
    for i in range(n_set):
        # set paths for storing channel config files
        train_path = './ischleseg/deepmedic/versions/DM_V1/configFiles/train_' + str(i)
        validation_path = './ischleseg/deepmedic/versions/DM_V1/configFiles/validation_' + str(i)

        if not os.path.exists(train_path): os.makedirs(train_path)
        if not os.path.exists(validation_path): os.makedirs(validation_path)

        # train and validation division point
        train_val_divison = int(np.floor(len(channels['Channels_ADC']) * 0.8))

        # random subject indices
        indices = range(len(channels['Channels_ADC']))
        random.shuffle(indices)
        
        for name, files in channels.iteritems():
            # save train channel files
            data_to_file([files[i] for i in indices[:train_val_divison]], os.path.join(train_path, 'train' + name + '.cfg'))
            # save validation channel files
            data_to_file([files[i] for i in indices[train_val_divison:]], os.path.join(validation_path, 'validation' + name + '.cfg'))


        # save names of predictions
        names = ['pred_ISLES2017_' + os.path.basename(x).split('.')[-3] for x in [files[i] for i in indices[train_val_divison:]]]
        data_to_file(names, os.path.join(validation_path, 'validationNamesOfPredictions.cfg'))
        
# modelConfig,cfg, trainConfig.cfg and testConfig.cfg must be added and modified manually.
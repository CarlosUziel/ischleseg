
# coding: utf-8

# In[6]:


#%% Imports
import os
import shutil
import nibabel as nib
import numpy as np
from random import shuffle
from glob import glob
from scipy import ndimage
from nilearn.image import resample_to_img, resample_img
from nilearn.masking import compute_background_mask, compute_epi_mask
from nilearn.plotting import plot_roi, plot_epi


# In[7]:


#%% Set current directory
os.chdir('/home/uziel/DISS')
test_flag = 1


# In[8]:


#%% List all sequences per subject
# linux
if test_flag:
    root = './data/ISLES2017/testing'
else:
    root = './data/ISLES2017/training'

subjects_paths = sorted(os.listdir(root))
channels_per_subject = dict() # groups relevant sequences per subject
for i in range(len(subjects_paths)):
    s_path = os.path.join(root, subjects_paths[i])
    channels_per_subject[i] = [y for x in os.walk(s_path) for y in
                        glob(os.path.join(x[0], '*ADC*.nii')) or
                        glob(os.path.join(x[0], '*MTT*.nii')) or
                        glob(os.path.join(x[0], '*rCBF*.nii')) or
                        glob(os.path.join(x[0], '*rCBV*.nii')) or
                        glob(os.path.join(x[0], '*Tmax*.nii')) or
                        glob(os.path.join(x[0], '*TTP*.nii')) or
                        glob(os.path.join(x[0], '*OT*.nii'))]


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


# In[9]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[10]:


# Generate files listing all images per channel
# linux
if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'
    
channels = {}
# channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
channels['Channels_ADC'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))]
channels['Channels_MTT'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))]
channels['Channels_rCBF'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))]
channels['Channels_rCBV'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))]
channels['Channels_Tmax'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))]
channels['Channels_TTP'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))]
# labels
channels['GtLabels'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                        for y in glob(os.path.join(x[0], '*OT*.nii.gz'))]
# masks
channels['RoiMasks'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                        for y in glob(os.path.join(x[0], 'mask.nii.gz'))]


# In[13]:


######################################
##### FILES FOR DM_V1 (BASELINE) #####
######################################

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
    # set paths for storing channel config files
    train_path = './ischleseg/deepmedic/versions/DM_V1/configFiles/train'
    validation_path = './ischleseg/deepmedic/versions/DM_V1/configFiles/validation'

    if not os.path.exists(train_path): os.makedirs(train_path)
    if not os.path.exists(validation_path): os.makedirs(validation_path)

    # train and validation division point
    train_val_divison = int(np.floor(len(channels['Channels_ADC']) * 0.8))

    for name, files in channels.iteritems():
        # save train channel files
        data_to_file(files[:train_val_divison], os.path.join(train_path, 'train' + name + '.cfg'))
        # save validation channel files
        data_to_file(files[train_val_divison:], os.path.join(validation_path, 'validation' + name + '.cfg'))


    # save names of predictions
    names = ['pred_ISLES2017_' + os.path.basename(x).split('.')[-3] for x in files[train_val_divison:]]
    data_to_file(names, os.path.join(validation_path, 'validationNamesOfPredictions.cfg'))


# In[31]:


################################
##### TEST RESULTING FILES #####
################################

if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'

channels = {}
# channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
channels['Channels_ADC'] = [y for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))]
channels['Channels_MTT'] = [y for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))]
channels['Channels_rCBF'] = [y for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))]
channels['Channels_rCBV'] = [y for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))]
channels['Channels_Tmax'] = [y for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))]
channels['Channels_TTP'] = [y for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))]
# labels
channels['GtLabels'] = [y for x in os.walk(root)
                        for y in glob(os.path.join(x[0], '*OT*.nii.gz'))]
# masks
channels['RoiMasks'] = [y for x in os.walk(root)
                        for y in glob(os.path.join(x[0], 'mask.nii.gz'))]

# take 5 random subjects and check their images
indices = range(len(channels['Channels_ADC']))
shuffle(indices)
indices = indices[:5]

for i in indices:
    # load a random channel of subject i
    channel = random.choice([x for x in channels.keys() if 'Mask' not in x and "Label" not in x])
    #img = nib.load(channels[channel][i])
    img = nib.load(channels[channel][i])
    mask = nib.load(channels['RoiMasks'][i])
    print('Subject: ' + str(i) + '. Channel: ' + str(channel) + '. Shape: ' + str(img.shape))
    plot_epi(img) # plot_epi(img, cut_coords=(0,0,0)) -> use to see co-registered channels per subject
    plot_roi(mask, img)


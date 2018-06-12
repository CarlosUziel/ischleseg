
# coding: utf-8

# In[1]:


# Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
from glob import glob


# In[2]:


# Set current directory
os.chdir('/home/he/carlos/DISS/ischleseg/deepmedic/versions/DM_V1/configFiles/train')


# In[3]:


# Load entries from training files
training_files_entries = [[line.strip() for line in open(x, 'r')]
                          for x in os.listdir('.')
                          if 'Config' not in x]


# In[4]:


# Assign entries to corresponding subject
subjects_entries = {i:[x[i] for x in training_files_entries]
                   for i in range(len(training_files_entries[0]))}


# In[5]:


# For each subjec, create n new ones (default=1),
# whose lesion region is randomly sampled.
clones_number = 1
for subject, entries in subjects_entries.items():
    channels = [x for x in entries if "OT" not in x and "mask" not in x]
    mask = [x for x in entries if "mask" in x]    
    label = [x for x in entries if "OT" in x]
    subject_path = os.path.dirname(channels[0])
    
    # create new clones
    for i in range(clones_number):
        label_img = nib.load(label[0])
        label_data = label_img.get_data()
        # create path to save clone data
        clone_path = os.path.join(subject_path, 'clone_V2_'+ str(i))
        if os.path.exists(clone_path): shutil.rmtree(clone_path)
        os.makedirs(clone_path)
        # create each clone channel
        for j in range(len(channels)):
            channel_img = nib.load(channels[j])
            channel_data = channel_img.get_data().copy()
            # get data withing roi (label)
            roi_data = channel_data[np.nonzero(label_data)]
            # new data follows gaussian distribution
            mean_value, std_value = [np.mean(roi_data), np.std(roi_data)]
            channel_data[np.nonzero(label_data)] = np.array([random.gauss(mean_value, std_value)
                                                             for _ in range(roi_data.shape[0])])
            # create modified channel for clone
            modified_channel = nib.Nifti1Image(channel_data, channel_img.affine)
            #TODO: Normalize image?
            # save clone channel
            channel_name = os.path.basename(channels[j])
            nib.save(modified_channel, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + channel_name))
        
        # save unaltered label for clone
        label_name = os.path.basename(label[0])
        nib.save(label_img, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + label_name))
        #save unaltered mask for clone
        mask_img = nib.load(mask[0])
        mask_name = os.path.basename(mask[0])
        nib.save(mask_img, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + mask_name))

    print("Subject " + str(subject) + " finished.")
            


# In[6]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[7]:


###############################################################
##### FILES FOR DM_V2 (BASELINE + RANDOM LESION SAMPLING) #####
###############################################################

# Get training subjects
subject_list = [os.path.split(os.path.dirname(x[0]))[1] for x in subjects_entries.values()]
subject_list.append('clone_V2_0') #add clone subdir name
#%% Generate files listing all images per channel
os.chdir('/home/he/carlos/DISS')
root = './data_processed/ISLES2017/training'

# copy configFiles from DM_V1 (baseline)
config_path = './ischleseg/deepmedic/versions/DM_V1/configFiles'
new_config_path = './ischleseg/deepmedic/versions/DM_V2/configFiles'
if os.path.exists(new_config_path): shutil.rmtree(new_config_path)
shutil.copytree(config_path, new_config_path)

channels = {}
# channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
channels['Channels_ADC'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
channels['Channels_MTT'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
channels['Channels_rCBF'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
channels['Channels_rCBV'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
channels['Channels_Tmax'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                             for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
channels['Channels_TTP'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
# labels
channels['GtLabels'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                        for y in glob(os.path.join(x[0], '*OT*.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]
# masks
channels['RoiMasks'] = [os.path.join('../../../../../../', y) for x in os.walk(root)
                        for y in glob(os.path.join(x[0], '*mask.nii.gz'))
                           if os.path.basename(x[0]) in subject_list]

train_path = './ischleseg/deepmedic/versions/DM_V2/configFiles/train'
for name, files in channels.iteritems():
    # save train channel files
    data_to_file(files, os.path.join(train_path, 'train' + name + '.cfg'))


from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256
width = 256
channels = 3

############################################################# Prepare  data set #################################################
train_slices = '../Dataset/training_data/training_data_bmp/slices/'
train_masks = '../Dataset/training_data/training_data_bmp/masks/'

train_x_list = glob.glob(train_slices + '/*.bmp')
train_y_list = glob.glob(train_masks + '/*.bmp')

train_x = np.zeros([5000, height, width, channels])
train_y = np.zeros([5000, height, width])

print('Reading')
for idx in range(5000):
    print(idx + 1)
    img = sc.imread(train_x_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
    train_x[idx, :, :, :] = img

    mask = sc.imread(train_y_list[idx])
    mask = np.double(sc.imresize(mask, [height, width], interp='bilinear'))
    train_y[idx, :, :] = mask

print('Reading  finished')

################################################################ Make the train and test sets ########################################
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img = train_x[0:4000, :, :, :]
Validation_img = train_x[4000:4000 + 400, :, :, :]
Test_img = train_x[4000+400:5000, :, :, :]

Train_mask = train_y[0:4000, :, :]
Validation_mask = train_y[4000:4000 + 400, :, :, :]
Test_mask = train_y[4000+400:5000, :, :, :]

np.save('data_train', Train_img)
np.save('data_test', Test_img)
np.save('data_val', Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test', Test_mask)
np.save('mask_val', Validation_mask)

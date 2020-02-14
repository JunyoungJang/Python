from __future__ import division, print_function, absolute_import

from skimage import io
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import os, tarfile, sys, urllib
from glob import glob
#########################################################################################
# (UTIL) If you don't have 5-model, It automatically setup on your folder.
DATA_URL = 'http://junyoungjang.ipdisk.co.kr:20925/publist/HDD1/EveryoneShare/model.tgz'
dest_directory = '.\\model'
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
filename = DATA_URL.split('/')[-1]
filepath = os.path.join(dest_directory, filename)
if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)  # tarfile extract

# Evaluating picture function
def cat_dog(result):
    index = result.index(max(result))
    prob = max(result)
    if index == 0:
        str = 'cat'
    elif index == 1:
        str = 'dog'
    return str, index, prob


def election_cat_dog(result):
    if result < 3:
        str = 'cat'
        index = 1
    else:
        str = 'dog'
        index = 0
    return str, index
#########################################################################################

#########################################################################################
# Step 1. Code to read 50 images that have not been given.
size_image = 64
test_path = 'test\\'
test_path = os.path.join(test_path, '*')
test_files = sorted(glob(test_path))
test_numfiles = len(test_files)
X = np.zeros((test_numfiles, size_image, size_image, 3), dtype='float64')
print('[Step.1] Read pictures. #', test_numfiles)
count = 0
num = 0
for f in test_files:
    try:
        img = io.imread(f)
        if num < 2 and (count+1)%10 == 0:
            plt.imshow(img)
            #########################################################################################
            # Step2 . Code to output a few images in the images.
            title_str ="[Step.2] A multiple of 10 - (%d/%d), File Name : %s "%(count+1, test_numfiles, f)
            plt.title(title_str, loc='left')
            plt.show()
            num += 1
            #########################################################################################
        new_img = imresize(img, (size_image, size_image, 3))
        X[count] = np.array(new_img)
        count += 1
    except:
        continue
#########################################################################################
from load_model import fivemodel
result_storage_ind = 0
simple_model_data, alex_model_data, google_model_data, vgg_model_data, res_model_data = fivemodel()

#########################################################################################
# Step 3. Classified as 'dog' and 'cat', output sorted values in order.
print('\t rank    |   result \t\t|\t  simple net, alex net, google net, vgg net, res net\t\t\t| file name')
for ind in range(test_numfiles):
    simple_str, simple_ind, simple_prob = cat_dog(simple_model_data[ind])
    alex_str, alex_ind, alex_prob = cat_dog(alex_model_data[ind])
    google_str, google_ind, google_prob = cat_dog(google_model_data[ind])
    vgg_str, vgg_ind, vgg_prob = cat_dog(vgg_model_data[ind])
    res_str, res_ind, res_prob = cat_dog(res_model_data[ind])

    result_str, result_ind = election_cat_dog(simple_ind + alex_ind + google_ind + vgg_ind + res_ind)
    result_storage_ind += result_ind

    print('[Step.3]', "%02d"%(ind+1), ' | result : ', result_str, '\t|\t\t',
          simple_str, '\t,',
          alex_str, '\t  ,',
          google_str, '\t  ,',
          vgg_str, '   ,',
          res_str,
          '\t\t\t    | file name :', test_files[ind])
#########################################################################################
# Step 4. Calculate the ratio of files recognized as 'cats' among 50 images and output.
print('[Step.4] Cat ratio in total picture : ', "%.2f %%"%(result_storage_ind/test_numfiles*100))
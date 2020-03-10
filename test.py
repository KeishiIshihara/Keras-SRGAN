#!/usr/bin/env python
#title           :test.py
#description     :to test the model
#author          :Deepak Birla
#date            :2018/10/30
#usage           :python test.py --options
#python_version  :3.5.4 

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
from scipy.misc import imresize
import argparse

import Utils, Utils_model
from Utils_model import VGG_LOSS

# image_shape = (96,96,3)
np.random.seed(10)
image_shape = (256,256,3)

def test_model(input_hig_res, model, number_of_images, output_dir):

    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, 'jpg', number_of_images)
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)

def test_model_for_lr_images(input_dir, output_dir, epoch, model, model_domain, downscale_factor, interp):

    output_dir = os.path.join(output_dir, model_domain)
    os.makedirs(output_dir, exist_ok=True)

    train_path, test_path, monitor_paths = Utils.load_dir({'input_dir': input_dir})

    monitoring_images_faces = Utils.load_data_from_dirs(monitor_paths[0], None)
    monitoring_images_landscapes = Utils.load_data_from_dirs(monitor_paths[1], None)
    monitoring_images = np.concatenate([monitoring_images_faces, monitoring_images_landscapes])

    print('[INFO] monitoring images {} found.'.format(monitoring_images.shape[0]))
    print('       shape:', monitoring_images.shape)

    Utils.plot_generated_images_for_monitoring(output_dir, 'trained_on_{}_epoch{}'.format(model_domain, epoch), model, monitoring_images, downscale_factor, interp)

    print('[INFO] finish testing.')




if __name__== "__main__":
    """test.py

    # Usage

    ```console
    $  CUDA_VISIBLE_DEVICES= python test.py -d='landscape' -e 35
    ```

    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                    default='/media/ssd1/srgan_dataset/dataset/',
                    help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir',
                    default='/media/ssd1/srgan_dataset/output_retesting/',
                    help='Path for Output images')

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=25 ,
                    help='Number of Images', type=int)

    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_lr_images',
                    help='Option to test model output or to test low resolution image')

    parser.add_argument('-ip', '--interpolation', action='store', dest='interpolation',
                    help='Which interp to use.', default='bicubic', type=str)

    parser.add_argument('-f', '--downscale_factor', action='store', dest='downscale_factor', default=4,
                    help='Downsampling scale factor', type= int)

    parser.add_argument('-d', '--test_domain', action='store', dest='test_domain', default='face',
                    choices=['face', 'landscape', 'both'],
                    help='Testing domain which is used when training the model to test', type= str)

    parser.add_argument('-e', '--epoch', action='store', dest='epoch', default=np.inf, type=int,
                    help='')


    values = parser.parse_args()


    loss = VGG_LOSS(image_shape)


    print('[INFO] setting up..')

    if values.epoch < np.inf:
        if values.test_domain == 'face':
            model_path = '/media/ssd1/srgan_dataset/model/faces/faces_gen_model{}.h5'.format(values.epoch)
            model = load_model(model_path , custom_objects={'vgg_loss': loss.vgg_loss})
            model_domain = 'face'
        elif values.test_domain == 'landscape':
            model_path = '/media/ssd1/srgan_dataset/model/landscapes/landscapes_gen_model{}.h5'.format(values.epoch)
            model = load_model(model_path , custom_objects={'vgg_loss': loss.vgg_loss})
            model_domain = 'landscape'
        elif values.test_domain == 'both':
            raise NotImplementedError

            model_path = '/media/ssd1/srgan_dataset/model/faces/both_gen_model50.h5'
            model = load_model(model_path , custom_objects={'vgg_loss': loss.vgg_loss})
            model_domain = 'both'
        else:
            raise ValueError('No such domain exist:', values.test_domain)
    else:
        raise ValueError('epoch must be an interger value.')

    print('[INFO] generator model is loaded.')
    print('[INFO] model_domain:', model_domain)
    print('[INFO] model chechpoint epoch:', values.epoch)

    # comfirmation
    if input('Do you want to contine? [y/enter/N]:').lower() in {'yes', 'y', 'ye', ''}:
        print('[INFO] Now start testing...')
    else:
        raise KeyboardInterrupt('Cancelled by user. Bye!')

    # main here
    if values.test_type == 'test_model': # high reso input
        test_model(values.input_hig_res, model, values.number_of_images, values.output_dir)

    elif values.test_type == 'test_lr_images': # low reso input
        test_model_for_lr_images(values.input_dir, values.output_dir, values.epoch, model, model_domain, values.downscale_factor, values.interpolation)

    else:
        print("No such option")





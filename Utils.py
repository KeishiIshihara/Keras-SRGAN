#!/usr/bin/env python
#title           :Utils.py
#description     :Have helper functions to process images and plot images
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr

# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale=None, interp='bicubic'):
    if downscale is not None:
        images = []
        for img in  range(len(images_real)):
            images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp=interp, mode=None))
        images_lr = array(images)
    else:
        images_lr = array(images_real)
    return images_lr

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5


def normalized_lr_images(input_data, downsample_factor, interp='bicubic'):

    return normalize(lr_images(input_data, downsample_factor, interp))


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def load_path(path):
    """
    load path to data by searching recursively
    """
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = data.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files

def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


def load_training_data(directory, ddomain='faces', number_of_images = 1000, train_test_ratio = 0.8, downscale_factor=4, interp='bicubic'):
    """load_training_data
    This will use only ground truth data and input data will be created by applying downsampling with given factor

    # Usage
    load face dataset:
        x_train_lr, x_train_hr, x_test_lr, x_test_hr = load_training_data(
                directory='/path/to/dataset/', ddomain='faces',
                number_of_images=2688, train_test_ratio=0.8,
                downsample_factor=4
        )
    """


    number_of_train_images = int(number_of_images * train_test_ratio)


    if not os.path.exists(directory):
        raise ValueError('directory path not found.')

    if ddomain == 'faces':
        print('[INFO] now loading face images..')
        ground_truth_dir = os.path.join(directory, 'faces/gt')
        files = load_data_from_dirs(load_path(ground_truth_dir), '.png')
    elif ddomain == 'landscapes':
        print('[INFO] now loading landscape images..')
        ground_truth_dir = os.path.join(directory, 'landscapes/gt')
        files = load_data_from_dirs(load_path(ground_truth_dir), '.jpg')
    else: # ddomain == 'both':
        print('Not yet available.')
        sys.exit()

        ground_truth_dir = os.path.join(directory, 'faces/gt')
        files_faces = load_data_from_dirs(load_path(ground_truth_dir), '.png')

        ground_truth_dir = os.path.join(directory, 'landscapes/gt')
        files_landscapes = load_data_from_dirs(load_path(ground_truth_dir), '.png')

        files = files_faces + files_landscapes

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    # test_array = array(files)
    # if len(test_array.shape) < 3:
    #     print("Images are of not same shape")
    #     print("Please provide same shape images")
    #     sys.exit()

    # train test split
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]

    # high resolution dataset
    x_train_hr = hr_images(x_train) # returns ndarray type object
    x_train_hr = normalize(x_train_hr) # [-1, 1]

    # low resolution dataset
    x_train_lr = lr_images(x_train, downscale=downscale_factor, interp=interp) # returns downsampled images by given factor
    x_train_lr = normalize(x_train_lr)

    # high resolution dataset
    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)

    # low resolution dataset
    x_test_lr = lr_images(x_test, downscale=downscale_factor, interp=interp)
    x_test_lr = normalize(x_test_lr)

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_mishima_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8, downsample_factor=2):
    """load_mishima_data

    # Usage
    load face dataset:
        x_train_lr, x_train_hr, x_test_lr, x_test_hr = load_training_data(directory='/path/to/dataset/faces', ext='.png',
                                                                          number_of_images=2688, train_test_ratio=0.8,
                                                                          downsample_factor=4)
    """

    number_of_train_images = int(number_of_images * train_test_ratio)

    ground_truth_dir = os.path.join(directory, 'gt')
    input_data_dir = os.path.join(directory, 'low_resolution', str(downsample_factor))

    if not os.path.exists(ground_truth_dir):
        raise ValueError('ground truth data path not found.')

    if not os.path.exists(input_data_dir):
        raise ValueError('input data path not found.')


    files_hr = load_data_from_dirs(load_path(ground_truth_dir), ext)
    files_lr = load_data_from_dirs(load_path(input_data_dir), ext)

    if (len(files_hr) or len(files_lr)) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d or %d" % len(files_hr), len(files_lr))
        sys.exit()


    # train test split
    x_train_hr = files_hr[:number_of_train_images]
    x_train_lr = files_lr[:number_of_train_images]
    x_test_hr = files_hr[number_of_train_images:number_of_images]
    x_test_lr = files_lr[number_of_train_images:number_of_images]

    # high resolution dataset
    x_train_hr = hr_images(x_train_hr) # returns ndarray type object
    x_train_hr = normalize(x_train_hr) # [-1, 1]

    # low resolution dataset
    # x_train_lr = lr_images(x_train, 4) # returns downsampled images by given factor
    x_train_lr = lr_images(x_train_lr)
    x_train_lr = normalize(x_train_lr)

    # high resolution dataset
    x_test_hr = hr_images(x_test_hr)
    x_test_hr = normalize(x_test_hr)

    # low resolution dataset
    x_test_lr = lr_images(x_test_lr)
    x_test_lr = normalize(x_test_lr)

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, x_test_hr

def load_test_data(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr

# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):

    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_image_{}.png'.format(epoch)))

    #plt.show()

# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images_on_batch(output_dir, epoch, generator, data_gen , downsample_factor, interp, dim=(1, 3), figsize=(15, 5)):

    len(data_gen)
    value = randint(len(data_gen))
    x_test_hr = data_gen[value]

    value = randint(len(x_test_hr))

    image_batch_hr = x_test_hr
    image_batch_lr = normalized_lr_images(x_test_hr, downsample_factor, interp)

    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_image_{}.png'.format(epoch)))

    #plt.show()


# Plots and save generated images(in form LR, SR, HR) from model to test the model
# Save output for all images given for testing
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):

    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    for index in range(examples):

        plt.figure(figsize=figsize)

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_generated_image_{}.png'.format(index)))

        #plt.show()

# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):

    examples = x_test_lr.shape[0]
    image_batch_lr = denormalize(x_test_lr)
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)

    for index in range(examples):

        #plt.figure(figsize=figsize)

        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'high_res_result_image_{}.png'.format(index)))

        #plt.show()






#!/usr/bin/env python
#title           :train.py
#description     :to train the model
#author          :Deepak Birla
#date            :2018/10/30
#usage           :python train.py --options
#python_version  :3.5.4


# do not display warnings and deprecated messages
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation_wrapper
deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# # gpu config
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(visible_device_list='0', allow_growth=True))
# set_session(tf.Session(config=config))

# main imports
from Network import Generator, Discriminator
import Utils_model, Utils
from Utils_model import VGG_LOSS

from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, sys
import argparse


np.random.seed(10)
# Remember to change image shape if you are having different size of images
# image_shape = (384,384,3)
image_shape = (256,256,3)



# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan

# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(params):
    """
    Train SRGAN!
    """
    # get params
    epochs = params['epochs']
    batch_size = params['batch_size']
    output_dir = params['output_dir']
    model_save_dir = params['model_save_dir']
    downscale_factor = params['downscale_factor']

    # load datasets
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(params['input_dir'], params['data_domain'], params['number_of_images'], params['train_test_ratio'] )
    loss = VGG_LOSS(image_shape)

    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])

    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(os.path.join(model_save_dir, 'losses.txt') , 'w+')
    loss_file.close()

    gan_loss_best = [np.inf] * 3
    discriminator_loss_best = np.inf

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            # rand_nums = np.random.choice(np.arange(x_train_hr.shape[0]), size=batch_size, replace=False)

            # get batch data
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr) # get generator prediction

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2 # biased random noise for real
            fake_data_Y = np.random.random_sample(batch_size)*0.2 # biased random noise for fake

            # train discriminator
            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # get batch data
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            # train gan (do not train discriminator)
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2 # biased noise for y
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])


        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        # gan_loss = str(gan_loss)

        loss_file = open(os.path.join(model_save_dir, 'losses.txt') , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, str(gan_loss), discriminator_loss) )
        loss_file.close()

        if e == 1 or e % 5 == 0:
            Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 100 == 0:
            generator.save(os.path.join(model_save_dir, 'gen_model{}.h5'.format(e)))
            discriminator.save(os.path.join(model_save_dir, 'dis_model{}.h5'.format(e)))
        # if gan_loss < gan_loss_best:
        #     gan_loss_best = gan_loss
        #     discriminator.save(os.path.join(model_save_dir, 'dis_model{}_by_monitoring_gen_loss.h5'.format(e)))
        #     generator.save(os.path.join(model_save_dir, 'gen_model{}_by_monitoring_gen_loss.h5'.format(e)))
        if discriminator_loss < discriminator_loss_best:
            discriminator_loss_best = discriminator_loss
            discriminator.save(os.path.join(model_save_dir, 'dis_model{}_by_monitoring_dis_loss.h5'.format(e)))
            generator.save(os.path.join(model_save_dir, 'gen_model{}_by_monitoring_dis_loss.h5'.format(e)))


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train_on_generator(params):
    """
    Train SRGAN feeling refreshed!
    """
    # get params
    epochs = params['epochs']
    batch_size = params['batch_size']
    input_dir = params['input_dir']
    output_dir = params['output_dir']
    model_save_dir = params['model_save_dir']
    downscale_factor = params['downscale_factor']
    interp = params['interpolation']

    # load datasets
    # x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(params['input_dir'], params['data_domain'], params['number_of_images'], params['train_test_ratio'] )

    datagen = image.ImageDataGenerator(validation_split = 1. - params['train_test_ratio'])
    train_generator = datagen.flow_from_directory(
            input_dir,
            target_size=(image_shape[0], image_shape[1]),
            batch_size=batch_size,
            class_mode=None,
            shuffle=True,
            subset = "training"
        )

    val_generator = datagen.flow_from_directory(
            input_dir,
            target_size=(image_shape[0], image_shape[1]),
            batch_size=batch_size,
            class_mode=None,
            shuffle=True,
            subset = "validation"
        )

    loss = VGG_LOSS(image_shape)

    train_generator

    # batch_count = int(x_train_hr.shape[0] / batch_size)
    batch_count = len(train_generator)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])

    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(os.path.join(model_save_dir, 'losses.txt') , 'w+')
    loss_file.close()

    gan_loss_best = [np.inf] * 3
    discriminator_loss_best = np.inf

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):

            # --- step 1: Discriminator
            # get batch data
            image_batch_hr = train_generator[rand_num]
            image_batch_lr = Utils.normalized_lr_images(image_batch_hr, downscale_factor, interp)
            generated_images_sr = generator.predict(image_batch_lr) # get generator prediction

            # make label data
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2 # biased random noise for real
            fake_data_Y = np.random.random_sample(batch_size)*0.2 # biased random noise for fake

            # train discriminator
            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # --- step 2: GAN (Generator)
            # get batch data
            rand_num = np.random.randint(len(train_generator))
            image_batch_hr = train_generator[rand_num]
            image_batch_lr = Utils.normalized_lr_images(image_batch_hr, downscale_factor, interp)

            # make label data
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2 # biased noise for y

            # train gan (do not train discriminator)
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])


        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        # gan_loss = str(gan_loss)

        loss_file = open(os.path.join(model_save_dir, 'losses.txt') , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, str(gan_loss), discriminator_loss) )
        loss_file.close()

        if e == 1 or e % 5 == 0:
            # Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
            Utils.plot_generated_images_on_batch(output_dir, e, generator, val_generator)
        if e % 100 == 0:
            generator.save(os.path.join(model_save_dir, 'gen_model{}.h5'.format(e)))
            discriminator.save(os.path.join(model_save_dir, 'dis_model{}.h5'.format(e)))
        # if gan_loss < gan_loss_best:
        #     gan_loss_best = gan_loss
        #     discriminator.save(os.path.join(model_save_dir, 'dis_model{}_by_monitoring_gen_loss.h5'.format(e)))
        #     generator.save(os.path.join(model_save_dir, 'gen_model{}_by_monitoring_gen_loss.h5'.format(e)))
        if discriminator_loss < discriminator_loss_best:
            discriminator_loss_best = discriminator_loss
            discriminator.save(os.path.join(model_save_dir, 'dis_model{}_by_monitoring_dis_loss.h5'.format(e)))
            generator.save(os.path.join(model_save_dir, 'gen_model{}_by_monitoring_dis_loss.h5'.format(e)))




if __name__== "__main__":
    """ train.py

    # Usage

    Run below command to train model. Set parameters accordingly.
    ```terminal
    $ python train.py -d downsample -do faces --input_dir='./input' --output_dir='./output' --model_save_dir='./model' -b 128 -e 100 -n 2688 -r 0.8 -f 4
    ```

    If you do not want to use or have gpu, then run:
    ```terminal
    $ CUDA_VISIBLE_DEVICES= python train.py -d downsample -do faces --input_dir='./input' --output_dir='./output' --model_save_dir='./model' -b 128 -e 100 -n 2688 -r 0.8 -f 4
    ```

    (keishish version)
    ```terminal-in-keishish
    $ CUDA_VISIBLE_DEVICES= python train.py -d downsample -do faces -b 128 -e 100 -n 2688 -r 0.8 -f 4
    ```

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-downsample', action='store', dest='data_downsample',
                    choices=['downsample', 'use_mishima_data'],
                    default='downsample',
                    help='Which way to do downsampling.')

    parser.add_argument('-do', '--data-domain', action='store', dest='data_domain',
                    choices=['faces', 'landscapes', 'both'],
                    default='faces',
                    help='Which data domain to use.')

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                    # default='/media/ssd1/srgan_dataset/images/',
                    default='/media/ssd1/srgan_dataset/image_test/',
                    help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir',
                    default='/media/ssd1/srgan_dataset/output',
                    help='Path for output images')

    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir',
                    default='/media/ssd1/srgan_dataset/model',
                    help='Path for model')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=4,
                    help='Batch Size', type=int)

    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=10,
                    help='number of iteratios for trainig', type=int)

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=16 ,
                    help='Number of Images', type= int)

    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8 ,
                    help='Ratio of train and test Images', type=float)

    parser.add_argument('-f', '--downscale_factor', action='store', dest='downscale_factor', default=4,
                    help='Downsampling scale factor', type= int)

    parser.add_argument('-g', '--train_on_generator', action='store_true', dest='train_on_generator',
                    help='Train using ImageDataGenerator')

    parser.add_argument('-ip', '--interpolation', action='store', dest='interpolation',
                    help='Which interp to use.', default='bicubic', type=str)
    values = parser.parse_args()


    # check params
    params = vars(values)

    # train
    if values.data_downsample == 'downsample':
        if values.train_on_generator:
            train_on_generator(params)
        else:
            train(params)

    elif values.data_downsample == 'use_mishima_data':
        pass
        # train_using_mishima_data(values.epochs, values.batch_size, values.input_dir, values.output_dir, values.model_save_dir, values.number_of_images, values.train_test_ratio, values.data_domain, values.downscale_factor)

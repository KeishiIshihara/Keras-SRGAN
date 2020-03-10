

import os
import sys
import glob
import numpy as np
from skimage import data, io, filters
from PIL import Image


def load_dataset(dataset_base, format='.png'):
    faces = os.path.join(dataset_base, 'images_train', 'faces', 'train')
    landscapes = os.path.join(dataset_base, 'images_train', 'landscapes', 'train')
    both = os.path.join(dataset_base, 'images_train', 'both','train')

    os.makedirs(both, exist_ok=True)

    faces_array = np.array(os.listdir(faces))
    print(('[INFO] now loading face images..'))
    image_faces = []
    for path in faces_array:
        image_faces.append(data.imread(os.path.join(faces, path)))

    image_faces = np.array(image_faces)
    np.random.shuffle(image_faces)
    image_faces = image_faces[:len(image_faces)//2]
    print('[INFO] face images shape:',image_faces.shape)

    # sys.exit()

    _landscapes_array = glob.glob(os.path.join(landscapes, '*'))

    # only dir
    _landscapes_array = [l for l in _landscapes_array if os.path.isdir(l)]
    # only file
    image_landscapes = []
    print(('[INFO] now loading landscape images..'))
    for i, scene in enumerate(_landscapes_array):
        tmp = []
        print('scene',scene)
        for path in os.listdir(scene):
            # print(os.path.join(scene, path))
            tmp.append(data.imread(os.path.join(scene, path)))
        image_landscapes.append(tmp)

    image_landscapes = np.array(image_landscapes)

    new_image_landscapes = []
    print(('[INFO] now random choicing landscape images..'))

    # shuffle first
    for scene in image_landscapes:
        np.random.shuffle(scene)

    # then append
    counter = 0
    while len(new_image_landscapes) < len(image_faces):
        for i in range(4):
            new_image_landscapes.append(image_landscapes[i][counter])
        counter += 1
        # scene = np.random.choice(len(image_landscapes))
        # idx = np.random.choice(len(image_landscapes[scene]))
        # new_image_landscapes.append(image_landscapes[scene][idx])

    new_image_landscapes = np.array(new_image_landscapes)
    new_image_landscapes = new_image_landscapes[:len(image_faces)]

    print('[INFO] landscape images shape:',new_image_landscapes.shape)

    print('[INFO] Total:', image_faces.shape[0]+new_image_landscapes.shape[0])


    if input('Do you want to contine? [y/enter/N]:').lower() in {'yes', 'y', 'ye', ''}:
        print('[INFO] Now start saving...')
    else:
        raise KeyboardInterrupt('Cancelled by user. Bye!')

    for i, face in enumerate(image_faces):
        path = os.path.join(both, 'face_{:0>4}'.format(i))
        Image.fromarray(face.astype(np.uint8)).save(path+format)

    for i, scene in enumerate(new_image_landscapes):
        path = os.path.join(both, 'scene_{:0>4}'.format(i))
        Image.fromarray(scene.astype(np.uint8)).save(path+format)

    print('[INFO] finish saving.')


if __name__ == '__main__':

    base_dir = '/media/ssd1/srgan_dataset/dataset'
    load_dataset(base_dir)

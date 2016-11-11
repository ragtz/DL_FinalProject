import numpy as np
import argparse
import os

def loadImgs(path):
    files = os.listdir(path)
    imgs = []

    for f in files:
        img = imread(os.path.join(path, f), mode='I')
        imgs.append(img)

    return np.array(imgs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save images to numpy array')
    parser.add_argument('--path', metavar='PATH', required=True, help='Path to image directory')
    parser.add_argument('--filename', metavar='FILE', required=True, help='Name of npy file')

    args = parser.parse_args()
    path = args.path
    filename = args.filename

    imgs = loadImgs(path)
    np.save(os.join(path, filename), imgs)
    

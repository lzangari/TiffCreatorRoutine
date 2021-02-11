from __future__ import division

import os
import argparse

import numpy as np

from PIL import Image as pilimage
import dm3lib as dm3
import tia_reader
from tifffile import imsave
import singlepixel_1_4_fig
import singlepixel_final_all


def frame_count(img_name):
    
    img = pilimage.open(img_name)
    
    n = 1
    has_frames = True
    while has_frames:
    
        try:
            img.seek(n)
            n = n + 1
            
        except EOFError:
            has_frames = False
            img.close()
    
    return n
    
def read_multiframe(img_name):

    img = pilimage.open(img_name)
    nframes = frame_count(img_name)
    imgarray = np.zeros((nframes, img.size[0], img.size[1]))
    
    for i in range(nframes):
        
        imgarray[i, :, :] = np.array(img)
        img.seek(i)
        
    return imgarray

def process(img_name, img_width, img_heigth, num, total):

    print('Start stacking images')

    name = os.path.split(img_name)[-1]
    print('Processing {} ({}/{}) ...'.format(name, num, total))
    if img_name.endswith('.tif'):
        img = pilimage.open(img_name)
        imgarray = np.array(img)
        
    elif img_name.endswith('.dm3'):
        img = dm3.DM3(img_name)
        imgarray = img.imagedata
        
    elif img_name.endswith('.ser'):
        img = tia_reader.serReader(img_name)
        imgarray = img['imageData'][:,:,0].astype('float32')

    
    ww, hh = img_width, img_heigth
    bg_value = np.min(imgarray)
    mask = np.ones((ww, hh), dtype='uint16') * bg_value
    
    # import matplotlib.pyplot as pl
    # pl.imshow(imgarray, interpolation='none')
    # pl.show()
    
    x0, y0 = np.unravel_index(imgarray.argmax(), imgarray.shape)
    imgxmax, imgymax = imgarray.shape
    xmin, xmax = x0 - ww//2, x0 + ww//2
    ymin, ymax = y0 - ww//2, y0 + ww//2
    
    mask[0 - min(0, xmin): ww  - max(0, xmax - imgxmax), 0 - min(0, ymin): hh - max(0, ymax - imgymax)] = imgarray[max(0, xmin): min(xmax, imgxmax), max(0, ymin): min(ymax, imgymax)]
    
    return mask
    


def process_images(src_dir, img_width, img_heigth, output_path):
    print(src_dir)
    src_images = [os.path.join(src_dir, fname)
                  for fname in sorted(os.listdir(src_dir))
                  if fname.rsplit('.')[-1] in ['tif', 'dm3', 'ser'] ]

    dsc_images = [process(img, img_width, img_heigth, i, len(src_images))
                  for i, img in enumerate(src_images, start=1)]

    final_image = np.asarray([img for img in dsc_images if img is not None])
    filename = os.path.join(output_path, 'output_stack.tif')
    imsave(filename, final_image, imagej=True)
    #singlepixel_1_4_fig.run_me(filename)
    singlepixel_final_all.run_me(filename)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', help='path to the images source directory', required=True)
    parser.add_argument('--output', help='path to the output stack image', default='output_stack.tif')
    parser.add_argument('--size', type=int, default=64, help='image size')
    
    args = parser.parse_args()
    
    process_images(args.srcdir, args.size, args.size, args.output)


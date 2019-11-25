#!/usr/bin/python
import h5py
import numpy as np
# import PIL
# from PIL import Image
import time
import os

def makePNG(fname, num):
    print('making png')
    res = 512
    # tile_x = 32
    # tile_y = res // tile_x
    # width = tile_x * res
    # height = tile_y * res
    #sample = 3

    # fname = "/home/alex/storage/Data/Sasha/discharge.zoom/flds.tot.209";
    with h5py.File(fname, 'r') as f:
        data_e = f['dens'][:,:,:]
        data_i = f['densi'][:,:,:]
    print('Finished reading hdf data')
    dims = data_e.shape
    print(dims)
    # img = np.zeros((width, height), dtype = 'uint32')
    print("Finished initializing data")

    offset = (dims[0] - res) // 2
    # print(offset)
    start = time.time()
    #data1 = data_e[::sample, ::sample, ::sample]
    #data2 = data_i[::sample, ::sample, ::sample]
    data1 = data_e[offset:dims[0]-offset:2, offset:dims[1]-offset:2, offset:dims[2]-offset:2]
    data2 = data_i[offset:dims[0]-offset:2, offset:dims[1]-offset:2, offset:dims[2]-offset:2]
    # red = (np.minimum(data1*256./100., 255).astype('uint32') << 24)
    red = (np.minimum(data1*256./100., 255).astype('uint8'))
    print(red.shape)
    # blue = (np.minimum(data2*256./100., 255).astype('uint32') << 16)
    # green = (np.minimum((data1 - data2)*256./100., 255).astype('uint32') << 8)
    # alpha = (np.minimum((data1 + data2)*256./100., 255).astype('uint32'))
            # (np.minimum(data2*256./100., 255).astype('uint32') << 16) +
            # (np.minimum((data1 + data2)*256./100., 255).astype('uint32'))
    # img = (red + blue + green + alpha)
    img = red
    # assign pixel values
    # clamp2png(data_i, data_e, tile_x, tile_y, res, offset, img)
    end = time.time()
    print("clamp used", end - start)
    # img = Image.frombytes('RGBA', img.shape, img)
    # img.save('/home/alex/storage/js-vis/public/textures/test' + str(res) + '.png' , format='png',compress_level = 1)
    img.astype(np.uint8).tofile('./3d/flds%03d.dat' % num)

if __name__ == "__main__":
    print("In main")
    for i in range(95, 105, 2):
        print("Working on", i)
        fname = "/tigress/sashaph/pulsar-run/discharge.zoom/flds.tot.%03d" % i
        makePNG(fname, i)

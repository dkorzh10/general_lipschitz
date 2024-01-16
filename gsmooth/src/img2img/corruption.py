# -*- coding: utf-8 -*-

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
import math
from multiprocessing import Pool
from PIL import Image


import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


class Corruption(object):

    param_max = {'gaussian_blur':5,
                 'zoom_blur':0.5,
                 'motion_blur':[15,90],
                 'rotational_blur':10,
                 'defocus_blur':5,
                 'rotate':10,
                 'translate':0.1,
                 'contrast':1,
                 'pixelate':1,
                 'scaling':1.3,
                 }

    def __init__(self, args,co_type='gaussian_blur',add_noise=0.,noise_sd=1,apply=True,distribution='none',parallel=False):
        self.args = args
        self.type = co_type

        self.noise_std = noise_sd
        self.distr = distribution

        self.add_noise = add_noise
        if apply == False:
            self.type = 'none'


        if self.distr == 'none':
            self.noise_std = 1
            self._get_rdnoise = self._uniform_noise
        elif self.distr == "uniform":
            self._get_rdnoise = self._uniform_noise

        elif self.distr == "gaussian":
            self._get_rdnoise = self._gaussian_noise

        elif self.distr == "exp":
            self._get_rdnoise = self._exp_noise

        elif self.distr == "folded_gaussian":
            self._get_rdnoise = self._folded_gaussian_noise
#         elif self.distr == "lognormal":
#             self._get_rdnoise = self._lognormal_noise

        else:
            raise NotImplementedError


    def _exp_noise(self, n, std):
        return np.random.exponential(scale=std,size=n)



    def _uniform_noise(self, n, std):
        return std*np.random.rand(n)

#     def _lognormal_noise(self, n, std):
#         return np.exp(std*np.random.randn(n))

    def _gaussian_noise(self,n, std):
        return std*np.random.randn(n)


    def _folded_gaussian_noise(self, n, std):
        return std * np.abs(np.random.randn(n))

    def apply(self, img, params=None):


        # for gaussian blur, we use uniform distribution from 0~5 sigma to corrupt the data
        if self.type == 'none':
            img_c = img
            params = np.array([0])[np.newaxis, ...]

        elif self.type == 'gaussian_blur':

            params = np.sqrt(36*self._get_rdnoise(1, self.noise_std)+0.01) if params is None else params
            img_c = gaussian_blur(img.astype(np.uint8), params[0])
            params = np.array(params)[np.newaxis,...]


        elif self.type == 'zoom_blur':
            params = 0.5*self._get_rdnoise(1, self.noise_std)+1 if params is None else params
            img_c = zoom_blur(img, params[0])
            params = np.array(params)[np.newaxis,...]


        elif self.type == 'motion_blur':
            params = 8*self._get_rdnoise(2, self.noise_std) if params is None else params
            params_ = params.copy()
            params_[0] = np.ceil(np.sqrt(params[0]**2+params[1]**2)+0.001)
            params_[1] = np.arctan(params[1]/(params[0]+0.001))*180/np.pi -45

            img_c = motion_blur(img, params_)


        elif self.type == "rotational_blur":
            params = 20*self._get_rdnoise(1,self.noise_std)-10 if params is None else params
            img_c = rotational_blur(img, params[0])
            params = np.array(params)[np.newaxis,...]


        elif self.type == "zoom_blur":
            params = 0.5*np.abs(self._get_rdnoise(1, self.noise_std)) + 1 if params is None else params
            img_c = zoom_blur(img, params[0])
            params = np.array(params)[np.newaxis]

        elif self.type == "defocus_blur":
            params = np.abs(8*self._get_rdnoise(1, self.noise_std)) if params is None else params
            img_c = defocus_blur(img, params[0])
            params = np.array(params)[np.newaxis]

        elif self.type == 'rotate':

            params = 100*self._get_rdnoise(1, self.noise_std)-50 if params is None else params
            img_c = rotate_image(img, params[0])
            params = np.array(params)[np.newaxis,...]

        elif self.type == 'scaling':
            params = 0.3*self._get_rdnoise(1, self.noise_std)+1 if params is None else params
            img_c = resize(img, params[0])
            params = np.array(params)[np.newaxis,...]

        elif self.type == 'translate':
            params = self._get_rdnoise(2, self.noise_std) if params is None else params

            img_c = translate_image_reflect(img, params)



        elif self.type == 'contrast':
            params = self._get_rdnoise(2, self.noise_std) if params is None else params
            img_c = brightness_contrast(img, params)

        elif self.type == 'pixelate':
            #smaller more severe, 0~1, certify 0.5~1
            # params = 0.1*self._get_rdnoise(1)+0.1
            params = self.noise_std* np.array([0.02*np.random.randint(5,50)]) if params is None else params
            img_c = pixelate(img, params[0])
            params = np.array(params)[np.newaxis,...]

        elif self.type == 'jpeg':
            # params = np.random.randint(5,20)
            params =10 if params is None else params
            img_c = jpeg_compression(img, params)
            params = np.array(params)[np.newaxis, ...]

        elif self.type == 'gaussian_noise':
            # label = np.random.randint(0,self.args.n_bins)
            label = 9
            params = np.zeros(self.args.n_bins)
            params[label] = 1
            # transform into discrete bins
            img_c = gaussian_noise(img, 0.05)
            # img_c = img



        else:
            raise ValueError
        if self.add_noise>0.01:
            img_c = img_c + self._get_rdnoise(img_c.size, self.add_noise).reshape(img_c.shape).astype(np.float32)
            img_c = np.clip(img_c,0,1)
        return img_c, params



    def _apply(self, img):
        img = img.numpy()*255
        img_c, params = self.apply(img)

        return (img_c, params)


    def batchapply(self, img, batchsize):
        pool = Pool(processes=16)
        img = (img.numpy()*255).transpose([1,2,0])
        imgs = [img for _ in range(batchsize)]
        time0 = time.time()

        _imgs_c = pool.map(self.apply, imgs)
        _imgs_c = list(zip(*_imgs_c))
        # print(time.time()- time0)

        imgs_c = torch.Tensor(np.stack(_imgs_c[0],axis=0).transpose([0,3,1,2]))
        params = torch.Tensor(np.stack(_imgs_c[1],axis=0))

        return imgs_c, params







def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 10:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=(0,0), sigmaX=alias_blur)


# Tell Python about the C method
# wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
#                                               ctypes.c_double,  # radius
#                                               ctypes.c_double,  # sigma
#                                               ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
# class MotionImage(WandImage):
#     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
#         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


'''use degree not rad, positive means left rotation, img: H*W*C, [0,255]->[0,1]'''
def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    if len(result.shape) <3:
        result = result[...,np.newaxis]
    return np.clip(result/255, 0, 1)

'''param  0 is height 1 is width (must be float), img H*W*C [0,255]->[0,1], params are the number of pixels to translate'''
def translate_image(img, params):
    T = np.array([[1,0, params[1]],[0,1,params[0]]])
    result = cv2.warpAffine(img, T, img.shape[1::-1],flags=cv2.INTER_LINEAR)
    return np.clip(result/255, 0,1)

'''translation towards (up, right), param  0 is height 1 is width, img H*W*C [0,255]->[0,1]'''
def translate_image_reflect(img, params):
    h, w, c = img.shape
    nx, ny = round(params[0]), round(params[1])
    nx, ny = nx % h, ny % w
    out = np.zeros_like(img).astype(np.float32)
    if nx > 0 and ny > 0:
        out[-nx:, -ny:, :] = img[ :nx, :ny, :]
        out[-nx:, :-ny, :] = img[ :nx, ny:, :]
        out[:-nx, -ny:, :] = img[ nx:, :ny, :]
        out[:-nx, :-ny, :] = img[ nx:, ny:, :]
    elif ny > 0:
        out[:, -ny:, :] = img[ :, :ny, :]
        out[:, :-ny, :] = img[ :, ny:, :]
    elif nx > 0:
        out[ -nx:, :, :] = img[ :nx, :, :]
        out[ :-nx, :, :] = img[ nx:, :, :]
    else:
        out = img
    out = out / 255
    return out



def clipped_zoom(img, zoom_factor):
    h, w = img.shape[0], img.shape[1]

    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))
    top = (h - ch) // 2
    wtop = (w - cw) //2
    img = scizoom(img[top:top + ch, wtop:wtop + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    wtrim_top = (img.shape[1]-w)//2
    return img[trim_top:trim_top + h, wtrim_top:wtrim_top + w]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

#[0,1]->[0,1]
def gaussian_noise(x, severity):
    # c = [0.04, 0.06, .08, .09, .10][severity - 1]
    c = severity

    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# dim:  n*h*w*c [0,255] (uint8) or [0,1] -> [0,1]
def gaussian_blur(x, sigma=1):
    # c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x) , sigma=sigma, multichannel=True)
    return np.clip(x, 0, 1)


def motion_noise(image, degree=12, angle=45):
    image = np.array(image)

    # This generates a matrix of motion blur kernels at any angle. The greater the degree, the higher the blur.
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    # blurred = np.array(blurred, dtype=np.uint8)
    return blurred

#[0,255]->[0,1]
def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (1.6,4,4)][severity - 1]    #parameters to be verified

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1)



# s [0.7,1.3]
def resize(x, s):
    x = torch.from_numpy(np.transpose(x, [2,0,1]))
    c, h, w = x.shape
    cy, cx = float(h - 1) / 2.0, float(w - 1) / 2.0
    rows = torch.linspace(0.0, h - 1, steps=h)
    cols = torch.linspace(0.0, w - 1, steps=w)

    nys = (rows - cy) / s + cy
    nxs = (cols - cx) / s + cx

    nysl, nxsl = torch.floor(nys), torch.floor(nxs)
    nysr, nxsr = nysl + 1, nxsl + 1

    nysl = nysl.clamp(min=0, max=h - 1).type(torch.LongTensor)
    nxsl = nxsl.clamp(min=0, max=w - 1).type(torch.LongTensor)
    nysr = nysr.clamp(min=0, max=h - 1).type(torch.LongTensor)
    nxsr = nxsr.clamp(min=0, max=w - 1).type(torch.LongTensor)

    nyl_mat, nyr_mat, ny_mat = nysl.unsqueeze(1).repeat(1, w), nysr.unsqueeze(1).repeat(1, w), nys.unsqueeze(1).repeat(
        1, w)
    nxl_mat, nxr_mat, nx_mat = nxsl.repeat(h, 1), nxsr.repeat(h, 1), nxs.repeat(h, 1)

    nyl_arr, nyr_arr, nxl_arr, nxr_arr = nyl_mat.flatten(), nyr_mat.flatten(), nxl_mat.flatten(), nxr_mat.flatten()

    imgymin = max(math.ceil((1 - s) * cy), 0)
    imgymax = min(math.floor((1 - s) * cy + s * (h - 1)), h - 1)
    imgxmin = max(math.ceil((1 - s) * cx), 0)
    imgxmax = min(math.floor((1 - s) * cx + s * (h - 1)), w - 1)

    # Pll_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
    #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    # Plr_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
    #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    # Prl_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
    #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    # Prr_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
    #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)

    Pll = torch.gather(x.reshape(c, h * w), dim=1, index=(nxl_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
    Plr = torch.gather(x.reshape(c, h * w), dim=1, index=(nxr_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
    Prl = torch.gather(x.reshape(c, h * w), dim=1, index=(nxl_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)
    Prr = torch.gather(x.reshape(c, h * w), dim=1, index=(nxr_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)

    # print(torch.sum(torch.abs(Pll - Pll_old)))
    # print(torch.sum(torch.abs(Plr - Plr_old)))
    # print(torch.sum(torch.abs(Prl - Prl_old)))
    # print(torch.sum(torch.abs(Prr - Prr_old)))

    nxl_mat, nyl_mat = nxl_mat.type(torch.FloatTensor), nyl_mat.type(torch.FloatTensor)

    out = torch.zeros_like(x)
    out[:, imgymin: imgymax + 1, imgxmin: imgxmax + 1] = (
                                                                 (ny_mat - nyl_mat) * (nx_mat - nxl_mat) * Prr +
                                                                 (1.0 - ny_mat + nyl_mat) * (nx_mat - nxl_mat) * Plr +
                                                                 (ny_mat - nyl_mat) * (1.0 - nx_mat + nxl_mat) * Prl +
                                                                 (1.0 - ny_mat + nyl_mat) * (
                                                                             1.0 - nx_mat + nxl_mat) * Pll)[:,
                                                         imgymin: imgymax + 1, imgxmin: imgxmax + 1]

    out = out.numpy().astype(np.float32).transpose([1,2,0])/255
    return out


# [0,255]->[0,1]
def defocus_blur(x, params):
    # c = [(3, 0.5), (4, 0.5), (0.5, 0.5), (1, 0.5), (1.5, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=params, alias_blur=0.5)

    channels = []
    for d in range(x.shape[-1]):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1)


#input [0,255] ->[0,1]
def motion_blur(x, params):


    x = motion_noise(x,int(params[0]), angle=params[1])


    if x.ndim < 3:
        x = x[...,np.newaxis]
    return np.clip(x,0,255) /255

#severity should be more than 1, [0,255]->[0,1]
def zoom_blur(x, severity=1):

    e = 0.02
    c = np.arange(1,severity,e)
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1)


def rotational_blur(x, theta):
    out = np.zeros_like(x).astype(np.float32)
    N = 10
    for i in range(N):
        elem = rotate_image(x, i/10*theta)
        # elem = np.where(elem>0.01,elem, x / 255)
        out += elem

    x = (np.array(x)/255.).astype(np.float32)
    # out = np.where(out>0.01,out, N*x)
    out = (x + out) / (N+1)

    return np.clip(out, 0, 1)

#[0,255]->[0,1]
def fog(x, severity=1):
    c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(mapsize=int(2**(np.ceil(np.log2(x.shape[0])))),wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1)

#[0,255]->[0,1]
def frost(x, severity=1):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
    frost = frost[x_start:x_start + 32, y_start:y_start + 32][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255) /255


#[0,255]->[0,1]
def snow(x, severity=1):
    c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    # x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    # snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    snow_layer = (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8)
    snow_layer = motion_noise(snow_layer,c[4],angle=np.random.uniform(-135, -45))
    output = BytesIO()
    _, snow_layer = cv2.imencode('.png', snow_layer)
    # snow_layer = MotionImage(blob=output.getvalue())

    # x = motion_noise(x,c[4], angle=np.random.uniform(-135, -45))
    # x *= 255

    snow_layer = cv2.imdecode(np.fromstring(snow_layer, np.uint8),1)          # snow 0~1
    # snow_layer = np.array(snow_layer)[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 128)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 255) /255


#[0,255]->[0,1]
def spatter(x, severity=1):
    c = [(0.62,0.1,0.7,0.7,0.5,0),
         (0.65,0.1,0.8,0.7,0.5,0),
         (0.65,0.3,1,0.69,0.5,0),
         (0.65,0.1,0.7,0.69,0.6,1),
         (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR)
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1)


#[0,255]->[0,1]
def contrast(x, severity):
    # c = [.75, .5, .4, .3, 0.15][severity - 1]

    c = severity
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1)


def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255



# input [0,255]->[0,1]
def brightness_contrast(x, params):
    x = x /255
    x = np.exp(params[0])*(x + params[1])
    return x




def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    # c = [80, 65, 58, 50, 40][severity - 1]
    c = severity
    # output = BytesIO()
    _, encimg = cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,c])
    x = cv2.imdecode(encimg,1)
    x = x.astype(float)/255
    # x.save(output, 'JPEG', quality=c)
    # x = PILImage.open(output)

    return x

# [0,255]->[0,1]
def pixelate(x, severity=1.0):
    # c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    c = severity
    h, w = x.shape[0], x.shape[1]
    x = cv2.resize(x,(int(w * c), int(h * c)),interpolation=cv2.INTER_LINEAR)
    x = cv2.resize(x,(w, h), interpolation=cv2.INTER_LINEAR)
    x = x[...,np.newaxis] if x.ndim <3 else x
    return x/255


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5 [0,255]->[0,1]
def elastic_transform(image, severity=1):
    IMSIZE = image.shape[0]
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
         (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
         (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
         (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1)











# /////////////// End Distortions ///////////////

import collections
if __name__ == '__main__':
    print('Using CIFAR-10 data')

    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise
    d['Shot Noise'] = shot_noise
    d['Impulse Noise'] = impulse_noise
    d['Defocus Blur'] = defocus_blur
    d['Glass Blur'] = glass_blur
    d['Motion Blur'] = motion_blur
    d['Zoom Blur'] = zoom_blur
    d['Snow'] = snow
    d['Frost'] = frost
    d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Elastic'] = elastic_transform
    d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression

    d['Speckle Noise'] = speckle_noise
    d['Gaussian Blur'] = gaussian_blur
    d['Spatter'] = spatter
    d['Saturate'] = saturate


    test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False)
    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])


    for method_name in d.keys():
        print('Creating images for the corruption', method_name)
        cifar_c, labels = [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(test_data.data, test_data.targets):
                labels.append(label)
                cifar_c.append(np.uint8(corruption(convert_img(img))))

        np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/' + d[method_name].__name__ + '.npy',
                np.array(cifar_c).astype(np.uint8))

        np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/labels.npy',
                np.array(labels).astype(np.uint8))



import dicom
import scipy.ndimage
import os
import pandas as pd
import numpy as np
import math
import cv2

IMG_PX_SIZE = 150
HM_SLICES = 20
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def load_scan(path):
    try:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                    slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation -
                    slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices
    except:
        return None


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # These are outside of scan pixels.
    outside_image = image.min()
    image[image == outside_image] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(scan, image, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    batch = []
    for i in range(0, image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = image[i + j]
            img= cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(tmp)

    batch = np.array(batch)

    return batch


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def full_preprocess(scan):
    scan_pixels = get_pixels_hu(scan)
    resampled_img = resample(scan, scan_pixels, [1,1,1])
    resampled_img = normalize(resampled_img)
    resampled_img = zero_center(resampled_img)
    return resampled_img

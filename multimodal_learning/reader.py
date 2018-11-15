from __future__ import absolute_import
from __future__ import print_function

import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from dltk.core.io.preprocessing import *
from dltk.core.io.augmentation import *
from dltk.core.io.reader import AbstractReader


class OrgansReader(AbstractReader):
    """
    A custom reader for UK Biobank T1w structural brain MRI data with online preprocessing and augmentation

    """

    def __init__(self, dshape, name='lesion_reader'):
        dshapes = [list(dshape) + [1], list(dshape)]
        super(OrgansReader, self).__init__([tf.float32, tf.int32], dshapes, name)

    def _read_sample(self, id_queue, n_examples=1, is_training=True):
        """A read function for nii images using SimpleITK

        Parameters
        ----------
        id_queue : str
            a file name or path to be read by this custom function

        n_examples : int
            the number of examples to produce from the read image

        is_training : bool

        Returns
        -------
        list of np.arrays of image and label pairs as 5D tensors
        """

        img_fn = id_queue[0][0]
        lbl_fn = id_queue[0][1]
        
#        print('img',img_fn)
#        print('lbl',lbl_fn)
        im = sitk.ReadImage(img_fn)
        lab = sitk.ReadImage(lbl_fn)


        # Use a SimpleITK reader to load the multi channel nii images and labels for training
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(img_fn))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_fn))

        # Create a 4D image array with dimensions [x, y, z, channels] and preprocess them
        img = self._preprocess(t1)
        img = np.expand_dims(img, axis=3)

        # Augment if in training mode
        if is_training:
            img, lbl = self._augment(img, lbl, n_examples)
        else:
            # Insert a dummy batch dimension
            img = img[np.newaxis, :]
            lbl = lbl[np.newaxis, :]

        return [img, lbl]

    def _preprocess(self, data):
        """ Simple whitening """
        return [whitening(img) for img in data]

    def _augment(self, img, lbl, n_examples):
        """Data augmentation during training

        Parameters
        ----------
        img : np.array
            a 4D input image with dimensions [x, y, z, channels]

        lbl : np.array
            a 3D label map corresponding to img

        n_examples : int
            the number of examples to produce from the read image

        Returns
        -------
        list of np.arrays of image and label pairs as 5D tensors
        """

        # Extract training example image and label pairs

        imgs, lbls = extract_class_balanced_example_array(img, lbl, example_size=self.dshapes[0][:-1],
                                                          n_examples=n_examples, classes=5)

        # Add Gaussian noise and offset to the images
        imgs = gaussian_noise(imgs, sigma=0.1)
        imgs = gaussian_offset(imgs, sigma=0.1)
        return imgs, lbls
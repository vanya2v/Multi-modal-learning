from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


import reader
import numpy as np
import pandas as pd
import tensorflow as tf


import dltk.core.modules as modules
from dltk.models.segmentation.fcn import ResNetFCN
from dltk.core import metrics as metrics
from dltk.core.io.sliding_window import SlidingWindow
import SimpleITK as sitk

num_classes = 5
DSC_all = []

def infer(args):
    s = tf.Session()

    filenames = pd.read_csv(args.csv, dtype=str).as_matrix()

    inputs, outputs = ResNetFCN.load(args.model_path, s)

    r = reader.OrgansReader([tf.float32, tf.int32],[[None, None, None, 1], [None, None, None]]) #,name='val_queue')

    
    for f in filenames:

        x, y = r._read_sample([f], is_training=False)

        sw = SlidingWindow(x.shape[1:4], [64, 64, 64], striding=[64, 64, 64])

        # Allocate the prediction output and a counter for averaging probabilities
        y_prob = np.zeros(y.shape + (num_classes,))
        y_pred_count = np.zeros_like(y_prob)
        for slicer in sw:
            y_sw = s.run(outputs['y_prob'], feed_dict={inputs[0]: x[slicer]})
            y_prob[slicer] += y_sw
            y_pred_count[slicer] += 1

        y_prob /= y_pred_count
        
        y_ = np.argmax(y_prob, axis=-1)

        dscs = metrics.dice(y_, y, num_classes)
        
        print(f[0] + ';  mean DSC = {:.3f}\n\t'.format(np.mean(dscs[1:]))
              + ', '.join(['DSC {}: {:.3f}'.format(i, dsc) for i, dsc in enumerate(dscs)]))

        y_ = np.squeeze (y_, axis = 0)

        itk_prediction = sitk.GetImageFromArray(y_)
        ds = np.transpose(dscs)
        DSC_all.append(ds)

    np.save('DSC_MR.npy', DSC_all)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Malibo inference script')

    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--csv', default='val_MR_set1.csv')
   
    parser.add_argument('--model_path', '-p', default='MR_set1/saves')
    parser.add_argument('--output_path', '-o', default='MR_set1')

    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    infer(args)
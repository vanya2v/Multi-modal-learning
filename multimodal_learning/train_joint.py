from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import namedtuple

import reader
import numpy as np
import pandas as pd
import tensorflow as tf

import dltk.core.modules as modules
from dltk.models.segmentation.fcn import ResNetFCN
from dltk.core import metrics as metrics
#from dltk.core.io.sliding_window import SlidingWindow
from dltk.core.utils import sliding_window_segmentation_inference
import SimpleITK as sitk

# Training parameters
training_params = namedtuple('training_params',
                             'max_steps, batch_size, save_summary_sec, save_model_sec, steps_eval')
training_params.__new__.__defaults__ = (1e4, 16, 10, 600, 100) #TODO: 16 BS for sorbus, 40 for monal04 1e6
tps = training_params()

num_classes = 5
num_channels = 1

label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 20, 21, 28]

def resize_sitk(inp, reference):
    resampleSliceFilter = sitk.ResampleImageFilter()
    resampled = resampleSliceFilter.Execute(inp, reference.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                                            inp.GetOrigin(), reference.GetSpacing(), inp.GetDirection(),
                                            0, inp.GetPixelIDValue())
    return resampled


def validate(ops, session, supervisor, name, v_all=True):
    """
        Run an inference on a validation dataset

        Parameters
        ----------
        ops : dict
            a dictionary containing all validation ops

        session : tf.session object

        supervisor : tf.Supervisor object

        Returns
        -------
    """

    # Pick num_validation_examples datasets to validate on
    if v_all:
        num_validation_examples = len(ops['filenames'])
    else:
        num_validation_examples = 4

    val_idx = range(num_validation_examples)

    # Track loss and Dice similarity coefficients as validation metrics
    val_loss = []
    val_dscs = []
#    val_orig_dscs = []

    # Iterate through the datasets and perform a sliding window inference
    for f in ops['filenames'][val_idx]:

        # Read a validation image and label of arbitrary dimensions
        val_x, val_y = ops['read_func']([f])

#        pid = os.path.basename(f[-1]).split('_')[0]
        pid = 'Subj.' + f[0].split('p/')[1][:2]

        y_prob = sliding_window_segmentation_inference(session, [ops['y_prob']], {ops['x']: val_x}, batch_size=16)[0]

        y_ = np.argmax(y_prob, axis=-1)

        # Compute the performance metrics for the dataset
        dscs = metrics.dice(y_, val_y, num_classes)
        loss = metrics.crossentropy(y_prob, np.eye(num_classes)[val_y.astype(int)], logits=False)
#        print(pid + '; CE= {:.6f}; DSC: l1 = {:.3f}'.format(loss, dscs[1]))
        print(pid + '; CE= {:.6f}; DSC:  LVR = {:.3f}, SPL = {:.3f}, RKDN = {:.3f}, LKDN = {:.3f}'.format(loss, dscs[1], dscs[2], dscs[3],dscs[4]))  
        # Collect the metrics over the validation data
        val_loss = val_loss + [loss]
        val_dscs = val_dscs + [dscs]

    np.save(args.save_dscs, val_dscs) 
    mean_dscs = np.mean(val_dscs, axis=0)
    mean_loss = np.mean(val_loss, axis=0)

    print('Mean; CE= {:.6f}; DSC: l1 = {:.3f}'.format(mean_loss, mean_dscs[1]))

    # Add the last computed dataset as an numpy image summary
    img_summaries = [modules.image_summary(val_x[0], name + '_img'),
                     modules.image_summary(val_y[0, :, :, :, np.newaxis] / num_classes, name + '_lbl'),
                     modules.image_summary(y_[0, :, :, :, np.newaxis] / num_classes, name + '_pred')]

    metrics_summaries = [modules.scalar_summary(mean_loss, name + '/ce'),
                         modules.scalar_summary(mean_dscs.mean(), name + '/dsc'),
                         modules.scalar_summary(mean_dscs[1:].mean(), name + '/dsc_wo_bg'),
                        ] + [modules.scalar_summary(mean_dscs[i + 1], name + '/dsc_lbl{}'.format(i + 1))
                             for i in range(num_classes - 1)]

    val_summaries = img_summaries + metrics_summaries
    return val_summaries


def create_ops(net, mode='train'):
    is_training = True if (mode == 'train') else False

    ops = {}

    ops['x'] = tf.placeholder(tf.float32, shape=[None, 64, 64, 64, num_channels], name='x_placeholder')
    ops['y'] = tf.placeholder(tf.int32, shape=[None, 64, 64, 64], name='y_placeholder')

    ops['net'] = net(ops['x'], is_training=is_training)

    ops['y_'] = ops['net']['y_']
    ops['y_prob'] = ops['net']['y_prob']

    # Define and add losses
    ops['loss_dice'] = modules.dice_loss(ops['net']['logits'], ops['y'], num_classes, include_background=True, smooth=0.,
                                         name='{}/loss_dice'.format(mode), collections=['losses', '{}'.format(mode)])

    ops['loss_all'] = ops['loss_dice'] # + val_ops['loss_l2']

    if is_training:
        # Add image summaries for x, y, y_
        modules.image_summary(ops['x'], 'train_img', ['{}'.format(mode)])
        modules.image_summary(tf.expand_dims(tf.to_float(ops['y_']) / num_classes, axis=-1), 'train_pred',
                              ['{}'.format(mode)])
        modules.image_summary(tf.expand_dims(tf.to_float(ops['y']) / num_classes, axis=-1), 'train_lbl',
                              ['{}'.format(mode)])

        # Add scalar summaries of the loss
        modules.scalar_summary(ops['loss_all'], '{}/loss_all'.format(mode), collections=['losses', '{}'.format(mode)])

        output_hist = tf.summary.histogram('out/soft', ops['y_prob'])

        # Merge all tf summaries from collection 'training' and all MOVING_AVERAGE_VARIABLES (i.e. batch norm)
        ops['summaries'] = tf.summary.merge(
            [tf.summary.merge_all('{}'.format(mode)), ] + [tf.summary.histogram(var.name, var) for var in
                                                           net.get_variables(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)]
            + [output_hist, ])

        # Create a learning rate placeholder for scheduling and choose an optimisation
        # ops['lr'] = tf.placeholder(tf.float32)
        ops['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        ops['optimiser'] = tf.train.AdamOptimizer(0.001, epsilon=1e-5).minimize(ops['loss_all'],
                                                                               global_step=ops['global_step'])

    return ops


def train(args):
    """
        Complete training and validation script. Additionally, saves inference model, trained weights and summaries.

        Parameters
        ----------
        args : argparse.parser object
            contains all necessary command line arguments

        Returns
        -------
    """

    if not args.resume:
        os.system("rm -rf %s" % args.save_path)
        os.system("mkdir -p %s" % args.save_path)
    else:
        print('Resuming training')

    g = tf.Graph()
    with g.as_default():

        # Set a seed
        np.random.seed(1337)
        tf.set_random_seed(1337)

        # Build the network graph
        net = ResNetFCN(num_classes,
                        num_residual_units=3,
                        filters=[16, 32, 64, 128, 256],
                        strides=[[1, 1, 1],[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]])

        # I/O ops via a custom reader with queueing
        print('Loading training file names from %s' % args.train_csv)
        train_ops = create_ops(net, mode='train')

        train_ops['filenames'] = pd.read_csv(args.train_csv, dtype=object).as_matrix()
        train_ops['reader'] = reader.OrgansReader([64, 64, 64], name='train_queue')
        train_ops['inputs'] =  train_ops['reader'](
            train_ops['filenames'],
            batch_size=tps.batch_size, n_examples=32,
            min_queue_examples=tps.batch_size * 3,
            capacity=tps.batch_size * 8, num_readers=4)
        train_ops['read_func'] = lambda x: train_ops['reader']._read_sample(x, is_training=False)

        train_ops['filenames2'] = pd.read_csv(args.train_csv2, dtype=object).as_matrix()
        train_ops['reader2'] = reader.OrgansReader([64, 64, 64], name='train_queue2')   
        train_ops['inputs2'] =  train_ops['reader2'](
            train_ops['filenames2'],
            batch_size=tps.batch_size, n_examples=32,
            min_queue_examples=tps.batch_size * 3,
            capacity=tps.batch_size * 8, num_readers=4)
        train_ops['read_func2'] = lambda x: train_ops['reader2']._read_sample(x, is_training=False)
        print('Loading training file names from %s' % args.train_csv2)
        
        if args.run_validation:
            print('Loading validation file names from %s' % args.val_csv)
            val_ops = create_ops(net, mode='val')

            val_ops['filenames'] = pd.read_csv(args.val_csv, dtype=str).as_matrix()
            val_ops['reader'] = reader.OrgansReader([None, None, None],
                                                       name='val_queue')
            # Get the read function in mode is_training=False to prevent augmentation
            val_ops['read_func'] = lambda x: val_ops['reader']._read_sample(x, is_training=False)

        # Define and set up a training supervisor, handling queues and logging for tensorboard
        net.save_metagraph(os.path.join(args.save_path, 'saves'), is_training=False)
        sv = tf.train.Supervisor(logdir=args.save_path,
                                 is_chief=True,
                                 summary_op=None,
                                 save_summaries_secs=tps.save_summary_sec,
                                 save_model_secs=tps.save_model_sec,
                                 global_step=train_ops['global_step'])

        s = sv.prepare_or_wait_for_session(config=tf.ConfigProto())

        # Main training loop
        step = s.run(train_ops['global_step']) if args.resume else 0
        while not sv.should_stop():

            if  (step %2 == 0):
                x, y = s.run(train_ops['inputs'])

                feed_dict = {train_ops['x']: x, train_ops['y']: y}

                (_, train_loss, train_y_) = s.run([train_ops['optimiser'], train_ops['loss_all'],train_ops['y_']],feed_dict=feed_dict)
            

            
            else:
                x, y = s.run(train_ops['inputs2'])
                feed_dict = {train_ops['x']: x, train_ops['y']: y}

                (_, train_loss, train_y_) = s.run([train_ops['optimiser'], train_ops['loss_all'],train_ops['y_']],feed_dict=feed_dict)
#
            # Evaluation of training and validation data
            if step % tps.steps_eval == 0:

                # Save the complete model
                net.save_model(os.path.join(args.save_path, 'saves'), s)

                # Compute the loss and summaries and save them to tensorboard
                (train_loss, train_y_, train_summaries) = s.run([train_ops['loss_all'], train_ops['y_'],
                                                                 train_ops['summaries']], feed_dict=feed_dict)
                sv.summary_computed(s, train_summaries, global_step=step)

                print("\nEval step= {:d}".format(step))
                print("Train: Loss= {:.6f} {:.6f}".format(train_loss, 1 - np.mean(metrics.dice(train_y_, y, num_classes))))

                # Run inference on validation data and save results to tensorboard
                if args.run_validation:
                    val_summaries = validate(ops=val_ops, session=s, supervisor=sv, name='val')
                    [sv.summary_computed(s, v, global_step=step) for v in val_summaries]

#                    train_val_summaries = validate(ops=train_ops, session=s, supervisor=sv, name='train_val', v_all=False)
#                    [sv.summary_computed(s, v, global_step=step) for v in train_val_summaries]

            # Stopping condition
            if step >= tps.max_steps and tps.max_steps > 0:
                print('Run %d steps of %d steps - stopping now' % (step, tps.max_steps))
                net.save_model(os.path.join(args.save_path, 'saves'), s)
                break
            step += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Abdominal Lesion segmentation training')
    parser.add_argument('--run_validation', default=True)

    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--train_csv', default='train_MR_set1.csv')
    parser.add_argument('--train_csv2', default='train_CT_set1.csv')
    parser.add_argument('--val_csv', default='val_CTMR_set1.csv')

    parser.add_argument('--save_dscs', default='DSC_CTMR_set1.npy')

    parser.add_argument('--save_path', '-p', default='CTMR_set1')
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    train(args)


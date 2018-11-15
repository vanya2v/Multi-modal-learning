## Multi-modal learning
Multi-modal learning is needed as in clinical practice, different imaging modalities (MRI, CT, Ultrasound, etc.) may be used to acquired the same structures. In practice, it is often difficult to acquire sufficient training data of a certain imaging modality.  Multi-modal learning in MRI andn CT are investigated using dual-stream encoder-decoder architecture. All of our MRI and CT data are unpaired, which means they are obtained from different subjects and not registered to each other. 


If you use this multi-modal learning repository in your work please refer to this citation:


Vanya V. Valindria, Nick Pawlowski, Martin Rajchl, Ioannis Lavdas, Eric O. Aboagye, Andrea G. Rockall, Daniel Rueckert, Ben Glocker. Multi-Modal Learning from Unpaired Images: Application to Multi-Organ Segmentation in CT and MRI. IEEE Winter Conference on  Applications of Computer Vision (WACV),  2018, pp. 547-556. 


## Deep Learning Toolkit (DLTK)
The repository was developed using DLTK:
[![Gitter](https://badges.gitter.im/DLTK/DLTK.svg)](https://gitter.im/DLTK/DLTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Documentation
DLTK API and user guides can be found [here](https://dltk.github.io/)


### Installation
1. Install CUDA with cuDNN and add the path to ~/.bashrc:

```shell
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MY_CUDA_PATH/lib64; export LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MY_CUDA_PATHextras/CUPTI/lib64; export LD_LIBRARY_PATH
PATH=$PATH:MY_CUDA_PATH/bin; export PATH
CUDA_HOME=MY_CUDA_PATH; export CUDA_HOME
```


2. Setup a virtual environment and activate it:

```shell
virtualenv venv_tf1.1
source venv_tf1.1/bin/activate
```

3. Install all DLTK dependencies (including tensorflow) via pip:

```shell
cd $DLTK_SRC
pip install -e .
```

### Start playing

1. Training: 
train_individual.py : train a single network on either CT or MR data only 
train_joint.py      : train a single network on both CT and MR data
train_dualstream.py : train dual-stream networks (FCN based, four versions or encoder-decoder streams) on both CT and MR data

2. Testing:
infer.py	    : infer for individual and joint networks
infer_dualstream.py : infer on dualstream networks for CT and MR data

 

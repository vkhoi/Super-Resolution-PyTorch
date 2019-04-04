# ESPCN - Efficient Sub-Pixel Convolutional Neural Network
Implementation of ESPCN as described in ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf).

Pre-upsampling super-resolution (SR) methods (e.g. SRCNN) upsamples the input image and performs reconstruction in the high-resolution (HR) space. The authors of ESPCN shows that this approach is computationally expensive (since all mappings are done in HR space) and it doesn't add any necessary information to construct the HR output. On the other hand, ESPCN extracts features in low-resolution (LR) space, then uses an efficient sub-pixel convolution layer that learns to upsample the LR feature maps into HR image.

## Data preparation
The network is trained on the *T91* dataset (so that it can be compared with *T91* SRCNN; training on *Imagenet* will be done in the future) and validated on the *Set5* dataset. These datasets can be downloaded from [here](http://vllab.ucmerced.edu/wlai24/LapSRN). After the datasets have been downloaded and extracted, please go to *config.py* to change the directory of the training and validation set.

## Training
I use the default weight initialization of PyTorch and Adam optimizer for training. The learning rate starts at *1e-3* and divides by *2* everytime the training loss plateaus. Training stops when learning rate is less than *1e-5* or validation PSNR does not improve after 200 epochs.

## Evaluation
Performance of the network is evaluated using the conventional benchmark of this literature - PSNR metric. To ensure that we get the correct PSNR number that can be used to compare with other works, it is advised [here](https://github.com/twtygqyy/pytorch-LapSRN) that we should use these functions from MATLAB (psnr, rgb2ycbcr, ycbcr2rgb, etc.). Therefore, I pick out the code from the [SRCNN project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) for this part.

## Usage
### Training
```
usage: train.py [-h] --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
                [--use_new_lr USE_NEW_LR] [--n_epochs N_EPOCHS]
                [--checkpoints_dir CHECKPOINTS_DIR]
                [--reload_from RELOAD_FROM] [--start_epoch START_EPOCH]
                [--seed SEED] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batch_size          batch size for training
  --use_new_lr          lr to use, otherwise use from config.py
  --n_epochs            number of epochs to train
  --checkpoints_dir     directory to save checkpoints
  --reload_from         directory to checkpoint to resume training from
  --start_epoch         epoch number to start from
  --seed SEED           random seed
  --cuda                whether to use cuda
```
An example of training:
```
$ python train.py --upscale_factor 2 --batch_size 64 --n_epochs 1000 --checkpoints_dir checkpoints --cuda
```
Train an ESPCN network with upscaling factor 2, batchsize 64 for 1000 epochs using CUDA. Checkpoint after each epoch is saved to *checkpoints* folder. 

### Evaluate on an image set
First, run *inference_save_to_mat.py* to super-resolve all images in an image set and save the results to a *.mat file.
```
usage: inference_save_to_mat.py [-h] --image_dir IMAGE_DIR --model MODEL
                                --upscale_factor UPSCALE_FACTOR --output
                                OUTPUT [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir           directory to image set for evaluating PSNR
  --model MODEL         model file
  --upscale_factor      upscale factor
  --output OUTPUT       output *.mat file
  --cuda                whether to use cuda
```
Example:
```
$ python inference_save_to_mat.py --image_dir ../../datasets/super-resolution/Set5 --model trained_model/ESPCN_upscale_2.pth --upscale_factor 2 --output result.mat
```
Super-resolve all images in Set5 dataset using the trained ESPCN model with upscale factor 2.

Then, go to folder *matlab_eval* and modify *eval_psnr.m* so that it has the correct directories to the image set and the ESPCN results. Finally, run *eval_psnr.m* to get the PSNR results.

### Super-resolve an image
```
usage: super_resolve.py [-h] --model MODEL --upscale_factor UPSCALE_FACTOR
                        --input INPUT --output OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --model               model file
  --upscale_factor      upscale factor
  --input INPUT         input image to super resolve
  --output OUTPUT       where to save the output image
```
An example to super-resolve an image:
```
$ python super_resolve.py --model trained_model/ESPCN_upscale_2.pth --upscale_factor 2 --input inp.png --output out.png
```

## Experimental results
* *T91* ESPCN

| DataSet | x2 upscaling (PSNR) | x3 upscaling (PSNR) | x4 upscaling (PSNR) |
| ------- |:-------------------:|:-------------------:|:--------------------:
| Set5    | 36.53               | 32.54               |                     |
| Set14   | 32.17               | 29.02               |                     |

* *Imagenet* ESPCN

The trained models that achieve these results are put in folder *trained_model*.

## References
* ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
* [PyTorch example of super-resolution DNN](https://github.com/pytorch/examples/tree/master/super_resolution)
* [Dataset fron LapSRN project](http://vllab.ucmerced.edu/wlai24/LapSRN)
* [Evaluation code from SRCNN project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
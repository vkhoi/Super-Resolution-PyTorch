# ESPCN - Efficient Sub-Pixel Convolutional Neural Network
Implementation of ESPCN as described in ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf).

Pre-upsampling super-resolution (SR) methods (e.g. SRCNN) upsamples the input image and performs reconstruction in the high-resolution (HR) space. The authors of ESPCN shows that this approach is computationally expensive (since all mappings are done in HR space) and it doesn't add any necessary information to construct the HR output. On the other hand, ESPCN extracts features in low-resolution (LR) space, then uses an efficient sub-pixel convolution layer that learns to upsample the LR feature maps into HR image.

## Data preparation
The network is trained on the *T91* and *General100* dataset and validated on the *Set5*, *Set14*, *BSD100* dataset. These datasets can be downloaded from [here](http://vllab.ucmerced.edu/wlai24/LapSRN).

## Train configurations
Train configurations can be modified in the _configs/*.yaml_ files. Training logs are outputted as _tensorboard_ to the _runs/_ folder.

## Usage
### Training
Just pass the _*.yaml_ configuration file you want to train with.
```
usage: train.py [-h] --config CONFIG [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --config              path to training config file
  --cuda                whether to use cuda
```

### Evaluate checkpoint on an image set
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
  --img_channels        # of image channels (1 for Y, 3 for RGB)
  --output OUTPUT       output *.mat file
  --cuda                whether to use cuda
```
Example:
```
$ python inference_save_to_mat.py --img_dir ../../datasets/super-resolution/Set5 --model trained_models/espcn_Y_scale2_t91_general100_adam.pth --upscale_factor 2 --img_channels 1 --output result.mat
```
Super-resolve all images in Set5 dataset using our trained model with x2 scale and receives luminance  channel as input.

Then, go to folder *../matlab_eval* and modify *eval_psnr.m* so that it has the correct directories to the image set and the *.mat file results. Finally, run *eval_psnr.m* to get the PSNR.

### Super-resolve an image
```
usage: super_resolve.py [-h] --model MODEL --upscale_factor UPSCALE_FACTOR
                        --input INPUT --output OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --model               model file
  --upscale_factor      upscale factor
  --img_channels        # of image channels (1 for Y, 3 for RGB)
  --input INPUT         input image to super resolve
  --output OUTPUT       where to save the output image
```
An example to super-resolve an image:
```
$ python super_resolve.py --model trained_models/espcn_Y_scale2_t91_general100_adam.pth --upscale_factor 2 --img_channels 1 --input inp.png --output out.png
```

## Experimental results

| DataSet | x2 upscaling (PSNR) | x3 upscaling (PSNR) | x4 upscaling (PSNR) |
| ------- |:-------------------:|:-------------------:|:-------------------:|
| Set5    | 37.07               | 33.10 (33.13)       | 30.72 (30.90)       |
| Set14   | 32.60               | 29.40 (29.49)       | 27.59 (27.73)       |
| BSD100  | 31.55               | 28.54               | 27.00               |

The trained models that achieve these results are put in folder *trained_model*.

## References
* ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
* [Dataset fron LapSRN project](http://vllab.ucmerced.edu/wlai24/LapSRN)
* [Evaluation code from SRCNN project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
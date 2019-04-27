# FSRCNN - Accelerating the Super-Resolution Convolutional Neural Network
Implementation of FSRCNN as described in ["Accelerating the Super-Resolution Convolutional Neural Network" - Dong et al.](https://arxiv.org/pdf/1608.00367.pdf).

An hourglass-shaped CNN structure for fast image super-resolution.

## Data preparation
The network is trained on the *T91* and *General100* dataset and validated on the *Set5*, *Set14*, *BSD200* dataset. These datasets can be downloaded from [here](http://vllab.ucmerced.edu/wlai24/LapSRN).

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
$ python inference_save_to_mat.py --img_dir ../../datasets/super-resolution/Set5 --model trained_models/fsrcnn_ychannel_t91_general100_adam.pth --upscale_factor 2 --img_channels 1 --output result.mat
```
Super-resolve all images in Set5 dataset using our trained model with x2 scale and receives luminance  channel as input.

Then, go to folder *matlab_eval* and modify *eval_psnr.m* so that it has the correct directories to the image set and the *.mat file results. Finally, run *eval_psnr.m* to get the PSNR.

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
$ python super_resolve.py --model trained_models/fsrcnn_ychannel_t91_general100_adam.pth --upscale_factor 2 --img_channels 1 --input inp.png --output out.png
```

## Experimental results
Training scripts for these models are located in _configs/_ folder.

| DataSet | x2 upscaling (PSNR) | x3 upscaling (PSNR) | x4 upscaling (PSNR) |
| ------- |:-------------------:|:-------------------:|:-------------------:|
| Set5    | 37.04               |                     |                     |
| Set14   | 32.57               |                     |                     |
| BSD100  | 31.53               |                     |                     |

The trained models that achieve these results are put in folder *trained_models*.

## References
* ["Accelerating the Super-Resolution Convolutional Neural Network" - Dong et al.](https://arxiv.org/pdf/1608.00367.pdf)
* [Dataset fron LapSRN project](http://vllab.ucmerced.edu/wlai24/LapSRN)
* [Evaluation code from SRCNN project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
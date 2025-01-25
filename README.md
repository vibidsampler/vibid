# ViBiDSampler

## How to use
### Environment setting
**Python** 3.10.14 \
**Torch** 2.0.1 

Our source code relies on [generative-models](https://github.com/Stability-AI/generative-models). \
Follow the environment setting from the [generative-models](https://github.com/Stability-AI/generative-models).

### Pre-trained model
Download the Stable Video Diffusion (SVD-XT) weights from [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). \
Specify the path to the downloaded model in the ```ckpt_path``` field of ```scripts/sampling/configs/svd_xt.yaml```.

### Video interpolation
In order to inference, run:
```
python scripts/sampling/vibidsampler.py
```
+ The paths to the source frames should be specified using the flags ```input_start_path``` and ```input_end_path```.
+ You can adjust the ```fps_id``` (approximately between 6 and 24) according to the specific use case.


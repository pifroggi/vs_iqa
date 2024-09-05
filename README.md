# Image Quality Assessment (IQA) for Vapoursynth
Looks at a frame and computes a quality score, or compares a pair of frames and computes how close they match.  
Then saves the score in a frame property called "vs_iqa_score".  
Optionally replaces frames if quality is too low.

This uses [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main), which comes with a large variety of different quality assessment metrics.

## Requirements
* [pytorch](https://pytorch.org/)
* `pip install numpy`
* `pip install pyiqa && pip install -U setuptools`

## Usage

    from vs_iqa import vs_iqa
    clip = vs_iqa(clip, ref, metric="hyperiqa", fallback, thresh=0.5, thresh_mode="lower", device="cuda", debug=False)

__*`clip`*__  
Clip to assess the quality of. Must be in RGBS or RGBH format.

__*`ref`* (optional)__  
Reference clip. If used, clip will be compared against ref. If not used, clip quality will be assessed blindly.  
A higher score means better match.  
Must be same format and dimensions as clip.

__*`metric`*__  
The algorithm or AI model that computes the score.  
Refer to [this table](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md) for all available options and to see which metrics need a reference (FR) and which do not (NR). The Model names column would be the input here.

__*`fallback`* (optional)__   
Replacement clip if quality is too low/high. Must be same format and dimensions as clip.

__*`thresh`* (optional)__  
Score at which the current frame should be replaced with fallback. Different metrics can produce very different ranges. Use "debug=True" to get an idea.  
Does nothing if fallback is not used.

__*`thresh_mode`* (optional)__  
"lower" = If the score is lower than tresh, frame will be replaced with fallback. (Bad quality frames will be replaced)  
"higher" = If the score is higher than tresh, frame will be replaced with fallback. (Good quality frames will be replaced)  
Does nothing if fallback is not used.

__*`device`* (optional)__  
Possible values are "cpu", or "cuda" to use with an Nvidia GPU. Some metrics are already very fast on CPU, others benefit greatly from a GPU.
RGBH format will additionally increase speed and half vram usage if the GPU supports fp16. (Not tested with all metrics, it is recommended to check scores against RGBS format for the chosen metric.)

__*`debug`* (optional)__  
Overlays the score onto the frame.

## Tips & Troubleshooting
* Enums are at the top of the script if needed.
* If nothing seems to happen, the model is probably downloading. Some are multiple hundred mb.

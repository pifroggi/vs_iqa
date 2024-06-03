import vapoursynth as vs
import numpy as np
import torch
from pyiqa import create_metric

core = vs.core

def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    array = np.stack([np.asarray(frame[p]) for p in range(frame.format.num_planes)], axis=-1)
    array = np.maximum(0, np.minimum(array, 1))
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor

def vs_iqa(clip, ref=None, fallback=None, thresh=0.5, thresh_mode='lower', metric='hyperiqa', device='cpu', debug=False):
    metric_mode = 'FR' if ref is not None else 'NR'
    
    if clip.format.id != vs.RGBS:
        raise ValueError("Clip must be in RGBS format.")

    if ref:
        if clip.format.id != vs.RGBS:
            raise ValueError("Reference clip must be in RGBS format.")
        if clip.width != ref.width or clip.height != ref.height:
            raise ValueError("Clip and reference clip must have the same dimensions.")
    
    if fallback:
        if clip.format.id != vs.RGBS:
            raise ValueError("Fallback must be in RGBS format.")

    iqa_model = create_metric(metric.lower(), metric_mode=metric_mode, device=torch.device(device))

    def _evaluate_frame(n, f):
        clip_frame = clip.get_frame(n)
        clip_tensor = frame_to_tensor(clip_frame)
        
        if ref:
            ref_frame = ref.get_frame(n)
            ref_tensor = frame_to_tensor(ref_frame)
            score = iqa_model(clip_tensor, ref_tensor).cpu().item()
            text = f'{metric} score: {score:.6f}\nreference: yes'
        else:
            score = iqa_model(clip_tensor).cpu().item()
            text = f'{metric} score: {score:.6f}\nreference: no'
        
        if fallback is not None:
            if (thresh_mode == 'lower' and score < thresh) or (thresh_mode == 'higher' and score > thresh):
                output_clip = fallback
                fallback_text = 'fallback: yes'
            else:
                output_clip = clip
                fallback_text = 'fallback: no'
        else:
            output_clip = clip
            fallback_text = 'fallback: not provided'
        
        output_clip = core.std.SetFrameProp(output_clip, prop='vs_iqa_score', floatval=score)

        # overlay text on frame
        if debug:
            text = f'{text}\n{fallback_text}'
            return core.text.Text(output_clip, text, alignment=9, scale=1)
        else:
            return output_clip

    return core.std.FrameEval(clip, eval=_evaluate_frame, prop_src=[clip])
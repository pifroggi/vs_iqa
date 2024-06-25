import vapoursynth as vs
import numpy as np
import torch
from pyiqa import create_metric
from enum import Enum
from typing import Optional

core = vs.core

class Device(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'

class ThreshMode(Enum):
    LOWER = 'lower'
    HIGHER = 'higher'

class Metric(Enum):
    #with reference (FR) metrics alphabetically
    AHIQ = ('ahiq', 'FR')
    CKDN = ('ckdn', 'FR')
    CW_SSIM = ('cw_ssim', 'FR')
    DISTS = ('dists', 'FR')
    FSIM = ('fsim', 'FR')
    GMSD = ('gmsd', 'FR')
    LPIPS = ('lpips', 'FR')
    LPIPS_VGG = ('lpips-vgg', 'FR')
    MAD = ('mad', 'NR')
    MS_SSIM = ('ms_ssim', 'FR')
    NLPD = ('nlpd', 'FR')
    PIEAPP = ('pieapp', 'FR')
    PSNR = ('psnr', 'FR')
    PSNRY = ('psnry', 'FR')
    SSIM = ('ssim', 'FR')
    SSIMC = ('ssimc', 'FR')
    STLPIPS = ('stlpips', 'FR')
    STLPIPS_VGG = ('stlpips-vgg', 'FR')
    TOPIQ_FR = ('topiq_fr', 'FR')
    TOPIQ_FR_PIPAL = ('topiq_fr-pipal', 'FR')
    VIF = ('vif', 'FR')
    VSI = ('vsi', 'FR')

    #no reference (NR) metrics alphabetically
    ARNIQA = ('arniqa', 'NR')
    ARNIQA_CSIQ = ('arniqa-csiq', 'NR')
    ARNIQA_CLIVE = ('arniqa-clive', 'NR')
    ARNIQA_FLIVE = ('arniqa-flive', 'NR')
    ARNIQA_KADID = ('arniqa-kadid', 'NR')
    ARNIQA_KONIQ = ('arniqa-koniq', 'NR')
    ARNIQA_LIVE = ('arniqa-live', 'NR')
    ARNIQA_SPAQ = ('arniqa-spaq', 'NR')
    ARNIQA_TID = ('arniqa-tid', 'NR')
    BRISQUE = ('brisque', 'NR')
    CLIPIQA = ('clipiqa', 'NR')
    CLIPIQA_PLUS = ('clipiqa+', 'NR')
    CLIPIQA_PLUS_RN50_512 = ('clipiqa+_rn50_512', 'NR')
    CLIPIQA_PLUS_VITL14_512 = ('clipiqa+_vitL14_512', 'NR')
    CLIPSCORE = ('clipscore', 'NR')
    CNNIQA = ('cnniqa', 'NR')
    DBCNN = ('dbcnn', 'NR')
    ENTROPY = ('entropy', 'NR')
    FID = ('fid', 'NR')
    HYPERIQA = ('hyperiqa', 'NR')
    ILNIQE = ('ilniqe', 'NR')
    INCEPTION_SCORE = ('inception_score', 'NR')
    LAION_AES = ('laion_aes', 'NR')
    LIQE = ('liqe', 'NR')
    LIQE_MIX = ('liqe_mix', 'NR')
    MANIQA = ('maniqa', 'NR')
    MANIQA_KONIQ = ('maniqa-koniq', 'NR')
    MANIQA_KADID = ('maniqa-kadid', 'NR')
    MANIQA_PIPAL = ('maniqa-pipal', 'NR')
    MUSIQ = ('musiq', 'NR')
    MUSIQ_AVA = ('musiq-ava', 'NR')
    MUSIQ_KONIQ = ('musiq-koniq', 'NR')
    MUSIQ_PAQ2PIQ = ('musiq-paq2piq', 'NR')
    MUSIQ_SPAQ = ('musiq-spaq', 'NR')
    NIQE = ('niqe', 'NR')
    NIMA = ('nima', 'NR')
    NIMA_KONIQ = ('nima-koniq', 'NR')
    NIMA_SPAQ = ('nima-spaq', 'NR')
    NIMA_VGG16_AVA = ('nima-vgg16-ava', 'NR')
    NRQM = ('nrqm', 'NR')
    PAQ2PIQ = ('paq2piq', 'NR')
    Pi = ('pi', 'NR')
    QALIGN = ('qalign', 'NR')
    TRES = ('tres', 'NR')
    TRES_FLIVE = ('tres-flive', 'NR')
    TRES_KONIQ = ('tres-koniq', 'NR')
    TOPIQ_IAA = ('topiq_iaa', 'NR')
    TOPIQ_IAA_RES50 = ('topiq_iaa_res50', 'NR')
    TOPIQ_NR = ('topiq_nr', 'NR')
    TOPIQ_NR_FACE = ('topiq_nr-face', 'NR')
    TOPIQ_NR_FLIVE = ('topiq_nr-flive', 'NR')
    TOPIQ_NR_SPAQ = ('topiq_nr-spaq', 'NR')
    UNIQUE = ('unique', 'NR')
    URANKER = ('uranker', 'NR')

    def __init__(self, metric_name, metric_type):
        self.metric_name = metric_name
        self.metric_type = metric_type

    @classmethod
    def from_string(cls, metric_name: str):
        for metric in cls:
            if metric.metric_name == metric_name.lower():
                return metric
        return None #if metric string not in enums

def frame_to_tensor(frame: vs.VideoFrame, device: str) -> torch.Tensor:
    array = np.empty((frame.height, frame.width, 3), dtype=np.float32)
    for p in range(frame.format.num_planes):
        array[..., p] = np.asarray(frame[p], dtype=np.float32)
    tensor = torch.from_numpy(array).to(device)
    tensor.clamp_(0, 1)
    return tensor.permute(2, 0, 1).unsqueeze(0)

def vs_iqa(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    metric: Metric | str = Metric.HYPERIQA,
    fallback: Optional[vs.VideoNode] = None,
    thresh: float = 0.5,
    thresh_mode: ThreshMode | str = ThreshMode.LOWER,
    device: Device | str = Device.CPU,
    debug: bool = False
) -> vs.VideoNode:

    #check if enum, known string, or unknown string
    if isinstance(thresh_mode, ThreshMode):
        thresh_mode = thresh_mode.value
    if isinstance(device, Device):
        device = device.value
    if isinstance(metric, Metric):
        metric_name = metric.metric_name
        metric_mode = metric.metric_type
    else:
        metric_enum = Metric.from_string(metric)
        if metric_enum:
            metric_name = metric_enum.metric_name
            metric_mode = metric_enum.metric_type
        else:
            metric_name = metric.lower()
            metric_mode = 'NR' if ref is None else 'FR'

    #checks
    if ref and metric_mode == 'NR':
        raise ValueError(f"Metric '{metric_name}' does not use a reference clip (NR).")
    if not ref and metric_mode == 'FR':
        raise ValueError(f"Metric '{metric_name}' needs a reference clip (FR).")
    if clip.format.id != vs.RGBS:
        raise ValueError("Clip must be in RGBS format.")
    if ref:
        if ref.format.id != vs.RGBS:
            raise ValueError("Reference clip must be in RGBS format.")
        if clip.width != ref.width or clip.height != ref.height:
            raise ValueError("Clip and reference clip must have the same dimensions.")
    if fallback:
        if fallback.format.id != vs.RGBS:
            raise ValueError("Fallback must be in RGBS format.")
    if thresh_mode not in ['higher', 'lower']:
        raise ValueError("thresh_mode must be either 'higher' or 'lower'.")

    iqa_model = create_metric(metric_name, metric_mode=metric_mode, device=torch.device(device))
    lower_better_text = "lower is better" if iqa_model.lower_better else "higher is better"

    def _evaluate_frame(n, f):
        clip_tensor = frame_to_tensor(f, device)
        
        if ref:
            ref_frame = ref.get_frame(n)
            ref_tensor = frame_to_tensor(ref_frame, device)
            score = iqa_model(clip_tensor, ref_tensor).cpu().item()
            text = f'({lower_better_text}) {metric_name} score: {score:.6f}\nreference: yes'
        else:
            score = iqa_model(clip_tensor).cpu().item()
            text = f'({lower_better_text}) {metric_name} score: {score:.6f}\nreference: no'
        
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

        #overlay text on frame
        if debug:
            text = f'{text}\n{fallback_text}'
            return core.text.Text(output_clip, text, alignment=9, scale=1)
        else:
            return output_clip

    return core.std.FrameEval(clip, eval=_evaluate_frame, prop_src=[clip])

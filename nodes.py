import argparse
import folder_paths
import os
import ffmpeg
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
import av
import numpy as np
import cv2
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

#from configs.prompts.test_cases import TestCasesDict
from .src.models.pose_guider import PoseGuider
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.unet_3d import UNet3DConditionModel
from .src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from .src.utils.util import get_fps, read_frames, save_videos_grid, calculate_file_hash, get_sorted_dir_files_from_directory, get_audio, lazy_eval, hash_path, validate_path

from .src.utils.mp_utils  import LMKExtractor
from .src.utils.draw_util import FaceMeshVisualizer

supported_model_extensions = set(['.pt', '.pth', '.bin', '.safetensors'])

folder_paths.folder_names_and_paths["pretrained_model"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_model"),
    ],
    supported_model_extensions
)

animation_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/prompts/animation.yaml")
animation_config = OmegaConf.load(animation_config_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_extensions = ['jpg', 'jpeg', 'png', 'gif']

class PoseGenVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image_path": ("RefImage_Path",),
                "pose_video_path": ("PoseVideo_Path", ),
                "vae_path": ([animation_config.pretrained_vae_path],),
                "model": ([animation_config.pretrained_base_model_path],),
                "weight_dtype": (["fp16", "fp32"],),
                "motion_module_path": ([animation_config.motion_module_path],),
                "image_encoder_path": ([animation_config.image_encoder_path],),
                "denoising_unet_path": ([animation_config.denoising_unet_path],),
                "reference_unet_path": ([animation_config.reference_unet_path],),
                "pose_guider_path": ([animation_config.pose_guider_path],),                
            },
        }

    RETURN_TYPES = ("FILENAMES",)
    RETURN_NAMES = ("Pose2Video",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "pose_generate_video"

    def pose_generate_video(self, ref_image_path, pose_video_path, vae_path, model, weight_dtype, motion_module_path, image_encoder_path, denoising_unet_path, reference_unet_path, pose_guider_path):
        print(ref_image_path)
        if weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        vae_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), vae_path)
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)), model)
        motion_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), motion_module_path) 
        image_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_encoder_path) 
        denoising_unet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), denoising_unet_path) 
        reference_unet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), reference_unet_path)
        pose_guider_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pose_guider_path)
        
        vae = AutoencoderKL.from_pretrained(vae_path,).to(device, dtype=weight_dtype)       
        reference_unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet",).to(dtype=weight_dtype, device=device)        
        
        inference_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), animation_config.inference_config)
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(model, motion_module_path, subfolder="unet", unet_additional_kwargs=infer_config.unet_additional_kwargs,).to(dtype=weight_dtype, device=device)       
        
        pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device=device, dtype=weight_dtype) # not use cross attention
        
        image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(dtype=weight_dtype, device=device)
        
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        generator = torch.manual_seed(42)

        width, height = 512, 512
        cfg = 3.5
        
        # load pretrained weights
        denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False,)
        reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"),)
        pose_guider.load_state_dict(torch.load(pose_guider_path, map_location="cpu"),)
        
        pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype=weight_dtype)
        
        #date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        #save_dir_name = f"{time_str}--seed_{args.seed}-{width}x{height}"

        #save_dir = Path(f"output/{date_str}/{save_dir_name}")
        #save_dir.mkdir(exist_ok=True, parents=True)
        
        save_dir = folder_paths.get_output_directory()

        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False)   
        
        ref_name = Path(ref_image_path).stem
        pose_name = Path(pose_video_path).stem
        print(ref_name)
        print(ref_image_path)
        print(pose_video_path)
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        ref_image_np = cv2.resize(ref_image_np, (512, 512))
            
        face_result = lmk_extractor(ref_image_np)
        assert face_result is not None, "Can not detect a face in the reference image."
        lmks = face_result['lmks'].astype(np.float32)
        ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)        
        
        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        args_L = len(pose_images)
        for pose_image_pil in pose_images[: args_L]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_image_np = cv2.cvtColor(np.array(pose_image_pil), cv2.COLOR_RGB2BGR)
            pose_image_np = cv2.resize(pose_image_np,  (width, height))
            pose_list.append(pose_image_np)       
         
        pose_list = np.array(pose_list)
            
        video_length = len(pose_tensor_list)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
            0
        )  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0) 
        
        video = pipe(ref_image_pil, pose_list, ref_pose, width, height, video_length, 25, 3.5, generator=generator,).videos        
        
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
        save_path = f"{save_dir}/{ref_name}_{pose_name}_{height}x{width}_{int(cfg)}_{time_str}_noaudio.mp4"
        save_videos_grid(video, save_path, n_rows=3, fps=src_fps)        

        audio_output = os.path.join(save_dir, 'audio_from_video.aac')
        # extract audio
        ffmpeg.input(pose_video_path).output(audio_output, acodec='copy').run()
        # merge audio and video
        stream = ffmpeg.input(save_path)
        audio = ffmpeg.input(audio_output)
        ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac').run()
            
        os.remove(save_path)
        os.remove(audio_output)   
             
        return (save_path.replace('_noaudio.mp4', '.mp4'),)
        
class RefImagePath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image_path": ("STRING", {"default": "X://insert/path/image.png", "aniportrait_path_extensions": image_extensions}),
            },
        }

    CATEGORY = "AniPortrait 🎥Video"

    RETURN_TYPES = ("RefImage_Path",)
    RETURN_NAMES = ("ref_image_path",)
    FUNCTION = "load_ref_image"

    def load_ref_image(self, **kwargs):
        if kwargs['ref_image_path'] is None or validate_path(kwargs['ref_image_path']) != True:
            raise Exception("reference image path is not a valid path: " + kwargs['ref_image_path'])
        return load_reference_image(**kwargs)

    @classmethod
    def IS_CHANGED(s, ref_image_path, **kwargs):
        return hash_path(ref_image_path)

    @classmethod
    def VALIDATE_INPUTS(s, ref_image_path, **kwargs):
        return validate_path(ref_image_path, allow_none=True)

def load_reference_image(ref_image_path: str):
    print(ref_image_path)
    return (ref_image_path,) # if return only one node,has to add comma

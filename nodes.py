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
from tqdm import tqdm
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
from .src.utils.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool
from .src.audio_models.model import Audio2MeshModel
from .src.utils.audio_util import prepare_audio_feature
from .src.utils.mp_utils  import LMKExtractor
from .src.utils.draw_util import FaceMeshVisualizer
from .src.utils.pose_util import project_points, project_points_with_trans

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from einops import rearrange

supported_model_extensions = set(['.pt', '.pth', '.bin', '.safetensors'])

folder_paths.folder_names_and_paths["pretrained_model"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_model"),
    ],
    supported_model_extensions
)

animation_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/prompts/animation.yaml")
animation_config = OmegaConf.load(animation_config_path)

animation_audio_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/prompts/animation_audio.yaml")
audio_config = OmegaConf.load(animation_audio_config_path)

animation_facereenac_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/prompts/animation_facereenac.yaml")
animation_facereenac_config = OmegaConf.load(animation_facereenac_config_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_extensions = ['jpg', 'jpeg', 'png', 'gif']
audio_extensions = ['wav', 'mp3']

class PoseGenVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "pose_images": ("IMAGE", ),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "height": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "width": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "seed": ("INT", {"default": 42}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 25, "min":0, "max": 50, "step": 1}),
                "vae_path": ([animation_config.pretrained_vae_path],),
                "model": ([animation_config.pretrained_base_model_path],),
                "weight_dtype": (["fp16", "fp32"],),
                "accelerate": ("BOOLEAN", {"default": True}),
                "fi_step": ("INT", {"default": 3}),
                "motion_module_path": ([animation_config.motion_module_path],),
                "image_encoder_path": ([animation_config.image_encoder_path],),
                "denoising_unet_path": ([animation_config.denoising_unet_path],),
                "reference_unet_path": ([animation_config.reference_unet_path],),
                "pose_guider_path": ([animation_config.pose_guider_path],),             
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "pose_generate_video"

    def pose_generate_video(self, ref_image, pose_images, frame_count, height, width, seed, cfg, steps, vae_path, model, weight_dtype, accelerate, fi_step, motion_module_path, image_encoder_path, denoising_unet_path, reference_unet_path, pose_guider_path):
        
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

        generator = torch.manual_seed(seed)

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
        

        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False)   
        
        if accelerate:
            frame_inter_model = init_frame_interpolation_model()
        
        
        ref_image = torch.squeeze(ref_image, 0)
        ref_image_pil = (ref_image.numpy() * 255).astype(np.uint8)

        ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        ref_image_np = cv2.resize(ref_image_np, (height, width))
            
        face_result = lmk_extractor(ref_image_np)
        assert face_result is not None, "Can not detect a face in the reference image."
        lmks = face_result['lmks'].astype(np.float32)
        ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)        
        
        pose_list = []
        pose_tensor_list = []
        print(f"pose video has {frame_count} frames")
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        
        for pose_image_pil in pose_images[: frame_count]:
            pose_image_pil = (pose_image_pil.numpy() * 255).astype(np.uint8)
            pose_tensor_list.append(pose_transform(Image.fromarray(pose_image_pil)))
        sub_step = fi_step if accelerate else 1
        for pose_image_pil in pose_images[: frame_count: sub_step]:
            pose_image = (pose_image_pil.numpy() * 255).astype(np.uint8)
            pose_image_np = cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR)
            pose_image_np = cv2.resize(pose_image_np,  (width, height))
            pose_list.append(pose_image_np)       
         
        pose_list = np.array(pose_list)
            
        video_length = len(pose_list)
        
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0) 
        
        video = pipe(Image.fromarray(ref_image_pil), pose_list, ref_pose, width, height, video_length, steps, cfg, generator=generator,).videos        
        
        if accelerate:
            video = batch_images_interpolation_tool(video, frame_inter_model, inter_frames=fi_step-1)
           
        
        ref_image_tensor = pose_transform(Image.fromarray(ref_image_pil))  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
            0
        )  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video.shape[2]
        ) 
              
        #video = torch.cat([ref_image_tensor, pose_tensor[:,:,:video.shape[2]], video], dim=0)

        outputs = []
        video = rearrange(video, "b c t h w -> t b c h w")
        for x in video:
            x = torchvision.utils.make_grid(x, nrow=1)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            x = (x * 255).numpy().astype(np.uint8)
            x = Image.fromarray(x)
            outputs.append(x)     
            
        iterable = (x for x in outputs)
        images = torch.from_numpy(np.fromiter(iterable, np.dtype((np.float32, (height, width, 3))))) / 255.0      
        return (images,)

        
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
    return (ref_image_path,) # if return only one node,has to add comma
  
class AudioPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "X://insert/path/audio.wav", "aniportrait_path_extensions": audio_extensions}),
            },
        }

    CATEGORY = "AniPortrait 🎥Video"

    RETURN_TYPES = ("Audio_Path",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "load_audio"

    def load_audio(self, **kwargs):
        if kwargs['audio_path'] is None or validate_path(kwargs['audio_path']) != True:
            raise Exception("reference audio path is not a valid path: " + kwargs['audio_path'])
        return load_reference_audio(**kwargs)

    @classmethod
    def IS_CHANGED(s, audio_path, **kwargs):
        return hash_path(audio_path)

    @classmethod
    def VALIDATE_INPUTS(s, audio_path, **kwargs):
        return validate_path(audio_path, allow_none=True)

def load_reference_audio(audio_path: str):
    return (audio_path,) # if return only one node,has to add comma
      
class GenerateRefPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("AniPortrait_Video",),
            },
        }

    RETURN_TYPES = ("FILENAMES",)
    RETURN_NAMES = ("ref_pose_path",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "generate_ref_pose"
    
    def generate_ref_pose(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("reference video path is not a valid path: " + kwargs['video'])
        lmk_extractor = LMKExtractor()

        cap = cv2.VideoCapture(kwargs['video'])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  

        pbar = tqdm(range(total_frames), desc="processing ...")
    
        trans_mat_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)
            result = lmk_extractor(frame)
            trans_mat_list.append(result['trans_mat'].astype(np.float32))
        cap.release()

        trans_mat_arr = np.array(trans_mat_list)

        # compute delta pose
        trans_mat_inv_frame_0 = np.linalg.inv(trans_mat_arr[0])
        pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

        for i in range(pose_arr.shape[0]):
            pose_mat = trans_mat_inv_frame_0 @ trans_mat_arr[i]
            euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
            pose_arr[i, :3] =  euler_angles
            pose_arr[i, 3:6] =  translation_vector

        # interpolate to 30 fps
        new_fps = 30
        old_time = np.linspace(0, total_frames / fps, total_frames)
        new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

        pose_arr_interp = np.zeros((len(new_time), 6))
        for i in range(6):
            interp_func = interp1d(old_time, pose_arr[:, i])
            pose_arr_interp[:, i] = interp_func(new_time)

        pose_arr_smooth = smooth_pose_seq(pose_arr_interp)
        save_dir = folder_paths.get_output_directory()
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{time_str}_pose.npy")
        np.save(save_path, pose_arr_smooth)
        return(save_path,)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)   
        
def matrix_to_euler_and_translation(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles, translation_vector


def smooth_pose_seq(pose_seq, window_size=5):
    smoothed_pose_seq = np.zeros_like(pose_seq)

    for i in range(len(pose_seq)):
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq
        
class Audio2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image_path": ("RefImage_Path",),
                "height": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "width": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "frames": ("INT", {"default": 0, "min":0, "max": 9999, "step": 1}),
                "seed": ("INT", {"default": 42}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 25, "min":0, "max": 50, "step": 1}),
                "frame_per_second": ("INT", {"default": 30, "min":0, "max": 240, "step": 1}),
                "vae_path": ([audio_config.pretrained_vae_path],),
                "model": ([audio_config.pretrained_base_model_path],),
                "weight_dtype": (["fp16", "fp32"],),
                "accelerate": ("BOOLEAN", {"default": True}),
                "fi_step": ("INT", {"default": 3}),
                "motion_module_path": ([audio_config.motion_module_path],),
                "image_encoder_path": ([audio_config.image_encoder_path],),
                "denoising_unet_path": ([audio_config.denoising_unet_path],),
                "reference_unet_path": ([audio_config.reference_unet_path],),
                "pose_guider_path": ([audio_config.pose_guider_path],),    
                "save_output": ("BOOLEAN", {"default": True}),            
            },
            "optional": {
                "video": ("AniPortrait_Video",),
                "audio_path": ("Audio_Path",),
                "ref_pose_path": ("FILENAMES", ),
            },
        }

    RETURN_TYPES = ("ANIPORTRAIT_FILENAMES",)
    RETURN_NAMES = ("Audio2Video",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "audio_2_video"

    def audio_2_video(self, ref_image_path, height, width, frames, seed, cfg, steps, frame_per_second, vae_path, model, weight_dtype, accelerate, fi_step, motion_module_path, image_encoder_path, denoising_unet_path, reference_unet_path, pose_guider_path, save_output=True, audio_path=None, ref_pose_path=None, video=None):
        # get output information
        save_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        if audio_path:
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
            
            inference_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_config.audio_inference_config)

            audio_infer_config = OmegaConf.load(inference_config_path)      

            # prepare model
            a2m_model = Audio2MeshModel(audio_infer_config['a2m_model'])
            a2m_model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_infer_config['pretrained_model']['a2m_ckpt'])), strict=False)
            a2m_model.cuda().eval()
            
            vae = AutoencoderKL.from_pretrained(vae_path,).to(device, dtype=weight_dtype)
            reference_unet = UNet2DConditionModel.from_pretrained(model,subfolder="unet",).to(dtype=weight_dtype, device=device)

            inference_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_config.inference_config)
            infer_config = OmegaConf.load(inference_config_path)
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(model, motion_module_path, subfolder="unet", unet_additional_kwargs=infer_config.unet_additional_kwargs,).to(dtype=weight_dtype, device=device)
            
            pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device=device, dtype=weight_dtype) # not use cross attention

            image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(dtype=weight_dtype, device=device)

            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)
            
            generator = torch.manual_seed(seed)

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
            pipe = pipe.to("cuda", dtype=weight_dtype)

            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M%S")
            
            lmk_extractor = LMKExtractor()
            vis = FaceMeshVisualizer(forehead_edge=False)
                    
            if accelerate:
                frame_inter_model = init_frame_interpolation_model()
        
            ref_name = Path(ref_image_path).stem
            audio_name = Path(audio_path).stem
            
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
            ref_image_np = cv2.resize(ref_image_np, (height, width))
                
            face_result = lmk_extractor(ref_image_np)
            assert face_result is not None, "No face detected."
            lmks = face_result['lmks'].astype(np.float32)
            ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
                
                
            sample = prepare_audio_feature(audio_path, wav2vec_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_infer_config['a2m_model']['model_path']))
            sample['audio_feature'] = torch.from_numpy(sample['audio_feature']).float().cuda()
            sample['audio_feature'] = sample['audio_feature'].unsqueeze(0)
                
            # inference
            pred = a2m_model.infer(sample['audio_feature'], sample['seq_len'])
            pred = pred.squeeze().detach().cpu().numpy()
            pred = pred.reshape(pred.shape[0], -1, 3)
            pred = pred + face_result['lmks3d']
                
            pose_seq = np.load(ref_pose_path)
            mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
            cycled_pose_seq = np.tile(mirrored_pose_seq, (sample['seq_len'] // len(mirrored_pose_seq) + 1, 1))[:sample['seq_len']]

            # project 3D mesh to 2D landmark
            projected_vertices = project_points(pred, face_result['trans_mat'], cycled_pose_seq, [height, width])
            
            pose_images = []
            for i, verts in enumerate(projected_vertices):
                lmk_img = vis.draw_landmarks((width, height), verts, normed=False)
                pose_images.append(lmk_img)

            pose_list = []
            pose_tensor_list = []
            print(f"pose video has {len(pose_images)} frames, with {frame_per_second} fps")
            pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
            frame_length = len(pose_images) if frames==0 else frames
            for pose_image_np in pose_images[: frame_length]:
                pose_image_pil = Image.fromarray(cv2.cvtColor(pose_image_np, cv2.COLOR_BGR2RGB))
                pose_tensor_list.append(pose_transform(pose_image_pil))
            sub_step = fi_step if accelerate else 1
            for pose_image_np in pose_images[: frame_length: sub_step]:
                pose_image_np = cv2.resize(pose_image_np,  (width, height))
                pose_list.append(pose_image_np)
                
            pose_list = np.array(pose_list)
            
            video_length = len(pose_list)

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)
            pose_tensor = pose_tensor.unsqueeze(0)

            video = pipe(
                ref_image_pil,
                pose_list,
                ref_pose,
                width,
                height,
                video_length,
                steps,
                cfg,
                generator=generator,
            ).videos    
            
            if accelerate:
                video = batch_images_interpolation_tool(video, frame_inter_model, inter_frames=fi_step-1)

            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                0
            )  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video.shape[2]
            )

            video = torch.cat([ref_image_tensor, pose_tensor[:,:,:video.shape[2]], video], dim=0)
            
            save_path = f"{save_dir}/{ref_name}_{audio_name}_{height}x{width}_{int(cfg)}_{time_str}_noaudio.mp4"
            save_videos_grid(
                video,
                save_path,
                n_rows=3,
                fps=frame_per_second,
            )
                
            stream = ffmpeg.input(save_path)
            audio = ffmpeg.input(audio_path)
            ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac', shortest=None).run()
            os.remove(save_path)        

        else:
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
            reference_unet = UNet2DConditionModel.from_pretrained(model,subfolder="unet",).to(dtype=weight_dtype, device=device)

            inference_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), animation_facereenac_config.inference_config)
            infer_config = OmegaConf.load(inference_config_path)
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(model, motion_module_path, subfolder="unet", unet_additional_kwargs=infer_config.unet_additional_kwargs,).to(dtype=weight_dtype, device=device)
            
            pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device=device, dtype=weight_dtype) # not use cross attention

            image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(dtype=weight_dtype, device=device)

            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)
            
            generator = torch.manual_seed(seed)

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
            pipe = pipe.to("cuda", dtype=weight_dtype)

            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M%S")
            
            lmk_extractor = LMKExtractor()
            vis = FaceMeshVisualizer(forehead_edge=False)        
            
            if accelerate:
                frame_inter_model = init_frame_interpolation_model()
            
            ref_name = Path(ref_image_path).stem
            pose_name = Path(video).stem
            
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
            ref_image_np = cv2.resize(ref_image_np, (height, width))
                
            face_result = lmk_extractor(ref_image_np)
            assert face_result is not None, "Can not detect a face in the reference image."
            lmks = face_result['lmks'].astype(np.float32)
            ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
                
            source_images = read_frames(video)
            src_fps = get_fps(video)
            print(f"source video has {len(source_images)} frames, with {src_fps} fps")
            pose_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )      
                       
            step = 1
            if src_fps == 60:
                src_fps = 30
                step = 2
            
            pose_trans_list = []
            verts_list = []
            bs_list = []
            src_tensor_list = []
            frame_length = len(source_images) if frames==0 else frames*step
            for src_image_pil in source_images[: frame_length: step]:
                src_tensor_list.append(pose_transform(src_image_pil))
            sub_step = step*fi_step if accelerate else step
            for src_image_pil in source_images[: frame_length: sub_step]:
                src_img_np = cv2.cvtColor(np.array(src_image_pil), cv2.COLOR_RGB2BGR)
                frame_height, frame_width, _ = src_img_np.shape
                src_img_result = lmk_extractor(src_img_np)
                if src_img_result is None:
                    break
                pose_trans_list.append(src_img_result['trans_mat'])
                verts_list.append(src_img_result['lmks3d'])
                bs_list.append(src_img_result['bs'])
                
            pose_arr = np.array(pose_trans_list)
            verts_arr = np.array(verts_list)
            bs_arr = np.array(bs_list)
            min_bs_idx = np.argmin(bs_arr.sum(1))

            # face retarget
            verts_arr = verts_arr - verts_arr[min_bs_idx] + face_result['lmks3d']
            # project 3D mesh to 2D landmark
            projected_vertices = project_points_with_trans(verts_arr, pose_arr, [frame_height, frame_width])
            
            pose_list = []
            for i, verts in enumerate(projected_vertices):
                lmk_img = vis.draw_landmarks((frame_width, frame_height), verts, normed=False)
                pose_image_np = cv2.resize(lmk_img,  (width, height))
                pose_list.append(pose_image_np)
            
            pose_list = np.array(pose_list)
            
            video_length = len(pose_list)                
                

            src_tensor = torch.stack(src_tensor_list, dim=0)  # (f, c, h, w)
            src_tensor = src_tensor.transpose(0, 1)
            src_tensor = src_tensor.unsqueeze(0)
            
            video_gen = pipe(
                ref_image_pil,
                pose_list,
                ref_pose,
                width,
                height,
                video_length,
                steps,
                cfg,
                generator=generator,
            ).videos

            if accelerate:
                video_gen = batch_images_interpolation_tool(video_gen, frame_inter_model, inter_frames=fi_step-1)

            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                0
            )  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_gen.shape[2]
            )
            
            video_gen = torch.cat([ref_image_tensor, video_gen, src_tensor[:,:,:video_gen.shape[2]]], dim=0)
            save_path = f"{save_dir}/{ref_name}_{pose_name}_{height}x{width}_{int(cfg)}_{time_str}_noaudio.mp4"
            save_videos_grid(
                video_gen,
                save_path,
                n_rows=3,
                fps=src_fps if frame_per_second==0 else frame_per_second,
            )
            
            audio_output = 'audio_from_video.aac'
            # extract audio
            ffmpeg.input(video).output(audio_output, acodec='copy').run()
            # merge audio and video
            stream = ffmpeg.input(save_path)
            audio = ffmpeg.input(audio_output)
            ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac', shortest=None).run()
            
            os.remove(save_path)
            os.remove(audio_output)
        
        save_prune_path = save_path.replace('_noaudio.mp4', '.mp4')
        previews = [
            {
                "save_prune_path": save_prune_path,
                "type": "output" if save_output else "temp",
            }
        ]            
        return {"ui": {"previews": previews}, "result": ((save_output, save_prune_path),)}      

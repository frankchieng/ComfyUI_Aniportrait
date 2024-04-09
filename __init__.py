import folder_paths
import os
import ffmpeg
from PIL import Image
import cv2
from tqdm import tqdm
import re
import torch
from .nodes import PoseGenVideo, RefImagePath, GenerateRefPose, Audio2Video, AudioPath

from .src.utils.util import get_fps, read_frames, save_videos_from_pil, calculate_file_hash, get_sorted_dir_files_from_directory, get_audio, lazy_eval, hash_path, validate_path
import numpy as np
from .src.utils.draw_util import FaceMeshVisualizer
from .src.utils.mp_utils  import LMKExtractor

video_extensions = ['webm', 'mp4', 'mkv', 'gif']

class VideoGenPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "AniPortrait"}),
                "height": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "width": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "generate_pose_video"

    def generate_pose_video(self, image, filename_prefix, height, width):
        
        frames = (image.numpy() * 255).astype(np.uint8)
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False)

        kps_results = []
        for i, frame_pil in enumerate(tqdm(frames)):
            image_np = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            image_np = cv2.resize(image_np, (height, width))
            face_result = lmk_extractor(image_np)
            try:
                lmks = face_result['lmks'].astype(np.float32)
                pose_img = vis.draw_landmarks((image_np.shape[1], image_np.shape[0]), lmks, normed=True)
                pose_img = Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB))
            except:
                pose_img = kps_results[-1]
            
            kps_results.append(pose_img)
            
        iterable = (x for x in kps_results)
        images = torch.from_numpy(np.fromiter(iterable, np.dtype((np.float32, (height, width, 3)))))
        return (images,)


class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "X://insert/path/here.mp4", "aniportrait_path_extensions": video_extensions}),
            },
        }

    CATEGORY = "AniPortrait 🎥Video"

    RETURN_TYPES = ("AniPortrait_Video", "IMAGE", "Frame_per_second", "AniPortrait_Audio", )
    RETURN_NAMES = ("video", "frames", "frame_per_second", "audio",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_av(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)
        

def load_video_av(video: str):
    fps = get_fps(video)
    frames = read_frames(video)
    input_dir = folder_paths.get_output_directory()
    audio_output = os.path.join(input_dir, 'audio_from_video.aac')

    return (video, frames, fps, audio_output)
     
NODE_CLASS_MAPPINGS = {
    "AniPortrait_Video_Gen_Pose": VideoGenPose,
    "AniPortrait_LoadVideoPath": LoadVideoPath,
    "AniPortrait_Pose_Gen_Video": PoseGenVideo,
    "AniPortrait_Ref_Image_Path": RefImagePath,
    "AniPortrait_Generate_Ref_Pose": GenerateRefPose,
    "AniPortrait_Audio2Video": Audio2Video,
    "AniPortrait_Audio_Path": AudioPath,    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AniPortrait_Video_Gen_Pose": "Video MediaPipe Face Detection🎥AniPortrait",
    "AniPortrait_LoadVideoPath": "Load Video (Path) 🎥AniPortrait",
    "AniPortrait_Pose_Gen_Video": "Pose Generate Video 🎥AniPortrait",
    "AniPortrait_Ref_Image_Path": "Ref Image Path 🎥AniPortrait",
    "AniPortrait_Generate_Ref_Pose": "Generate Ref Pose 🎥AniPortrait",
    "AniPortrait_Audio2Video": "Audio Gen Video 🎥AniPortrait",   
    "AniPortrait_Audio_Path": "Audio Path 🎥AniPortrait",   
}

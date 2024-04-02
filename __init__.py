import folder_paths
import os
import ffmpeg
from PIL import Image
import cv2
from tqdm import tqdm
import re

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
                "video": ("AniPortrait_Video",),
                "frames": ("IMAGE",),
                "frame_per_second": ('Frame_per_second',),
                "filename_prefix": ("STRING", {"default": "AniPortrait"}),
                "height": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "width": ("INT", {"default": 512, "min": 0, "max": 1024, "step": 1}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AniPortrait_Audio",),
            },
        }

    RETURN_TYPES = ("PoseVideo_Path",)
    RETURN_NAMES = ("pose_video_path",)
    OUTPUT_NODE = True
    CATEGORY = "AniPortrait 🎥Video"
    FUNCTION = "generate_pose_video"

    def generate_pose_video(self, video, frames, frame_per_second, filename_prefix, save_output, height, width, audio):
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        # comfy counter workaround
        max_counter = 0
        # Loop through the existing files
        matcher = re.compile(f"{re.escape(filename_prefix)}_(\d+)\D*\.[a-zA-Z0-9]+")
        for existing_file in os.listdir(full_output_folder):
            # Check if the file matches the expected format
            match = matcher.fullmatch(existing_file)
            if match:
                # Extract the numeric portion of the filename
                file_counter = int(match.group(1))
                # Update the maximum counter value if necessary
                if file_counter > max_counter:
                    max_counter = file_counter

        # Increment the counter by 1 to get the next available value
        counter = max_counter + 1
        output_file = f"{filename_prefix}_{counter:05}_kps_noaudio.mp4"
        out_path = os.path.join(full_output_folder, output_file)
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

        #print(out_path.replace('_noaudio.mp4', '.mp4'))
        save_videos_from_pil(kps_results, out_path, fps=frame_per_second)
        # save audio file into input directory
        ffmpeg.input(video).output(audio, acodec='copy').run()
        stream = ffmpeg.input(out_path)
        audio_receive = ffmpeg.input(audio)

        ffmpeg.output(stream.video, audio_receive.audio, out_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac').run()
        os.remove(out_path)
        os.remove(audio)
        previews = [
            {
                "filename": out_path,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp"
            }
        ]
        return (out_path.replace('_noaudio.mp4', '.mp4'),)

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

import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
import hashlib

from typing import Iterable
import subprocess
import re

from .logger import logger

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps
    
def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode("utf-8")
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score


if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.env["VHS_FORCE_FFMPEG_PATH"]
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)


def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = directory.strip()
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    h = hashlib.sha256()
    b = bytearray(10*1024*1024) # read 10 megabytes at a time
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it if requested
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()


def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-v", "error", "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    return subprocess.run(args + ["-f", "wav", "-"],
                          stdout=subprocess.PIPE, check=True).stdout


def lazy_eval(func):
    class Cache:
        def __init__(self, func):
            self.res = None
            self.func = func
        def get(self):
            if self.res is None:
                self.res = self.func()
            return self.res
    cache = Cache(func)
    return lambda : cache.get()


def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_sequence(path):
    #Check if path is a valid ffmpeg sequence that points to at least one file
    (path, file) = os.path.split(path)
    if not os.path.isdir(path):
        return False
    match = re.search('%0?\d+d', file)
    if not match:
        return False
    seq = match.group()
    if seq == '%d':
        seq = '\\\\d+'
    else:
        seq = '\\\\d{%s}' % seq[1:-1]
    file_matcher = re.compile(re.sub('%0?\d+d', seq, file))
    for file in os.listdir(path):
        if file_matcher.fullmatch(file):
            return True
    return False

def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(path.strip("\""))


def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        return True if allow_url else "URLs are unsupported for this path"
    if not os.path.isfile(path.strip("\"")):
        return "Invalid file path: {}".format(path)
    return True

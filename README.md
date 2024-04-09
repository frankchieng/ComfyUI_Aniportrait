#### Updates:
① Implement the frame_interpolation to speed up generation

② Modify the current code and support chain with the [VHS nodes](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), i just found that comfyUI IMAGE type requires the torch float32 datatype, and AniPortrait heavily used numpy of image unit8 datatype,so i just changed my mind from my own image/video upload and generation nodes to the prevelance SOTA VHS image/video upload and video combined nodes,it WYSIWYG and inteactive well and instantly render the result
- ✅ [2024/04/09] raw video to pose video with reference image(aka self-driven)
- ✅ [2024/04/09] audio driven
- ✅ [2024/04/09] face reenacment
### audio driven combined with reference image and reference video
![audio2video](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/c498a62e-ffa0-4cbc-a39d-b7daec586001)
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/d619f8b6-32ca-4aa2-8b18-d0649e3243ff" muted="false"></video>
    </td>
</tr>
</table>

### raw video to pose video with reference image
![pose2video](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/882e3685-ee13-4798-9f90-d195d6595a97)
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/edaa9907-8720-4435-8529-405c96a2e66d" muted="false"></video>
    </td>
</tr>
</table>

### face reenacment
![face_reenacment](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/2bb488ee-ec20-44d1-a941-542b4e0be8bc)
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/931d7661-8c79-4747-be5b-b1d361b7e7e9" muted="false"></video>
    </td>
</tr>
</table>

This is unofficial implementation of AniPortrait in ComfyUI custom_node,cuz i have routine jobs,so i will update this project when i have time
> [Aniportrait_pose2video.json](https://github.com/frankchieng/ComfyUI_Aniportrait/blob/main/assets/pose2video_workflow.json)

> [Audio driven](https://github.com/frankchieng/ComfyUI_Aniportrait/blob/main/assets/audio2video_workflow.json)

> [face reenacment](https://github.com/frankchieng/ComfyUI_Aniportrait/blob/main/assets/face_reenacment_workflow.json)

you should run
```shell
git clone https://github.com/frankchieng/ComfyUI_Aniportrait.git
```
then run 
```shell
pip install -r requirements.txt
```
download the pretrained models
> [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

> [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)

> [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

> [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) 

download the weights:
> [denoising_unet.pth](https://huggingface.co/ZJYang/AniPortrait/tree/main)
> [reference_unet.pth](https://huggingface.co/ZJYang/AniPortrait/tree/main)
> [pose_guider.pth](https://huggingface.co/ZJYang/AniPortrait/tree/main)
> [motion_module.pth](https://huggingface.co/ZJYang/AniPortrait/tree/main)
> [audio2mesh.pt](https://huggingface.co/ZJYang/AniPortrait/tree/main)
> [film_net_fp16.pt](https://huggingface.co/ZJYang/AniPortrait/tree/main)
```text
./pretrained_model/
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5
|   |-- feature_extractor
|   |   `-- preprocessor_config.json
|   |-- model_index.json
|   |-- unet
|   |   |-- config.json
|   |   `-- diffusion_pytorch_model.bin
|   `-- v1-inference.yaml
|-- wav2vec2-base-960h
|   |-- config.json
|   |-- feature_extractor_config.json
|   |-- preprocessor_config.json
|   |-- pytorch_model.bin
|   |-- README.md
|   |-- special_tokens_map.json
|   |-- tokenizer_config.json
|   `-- vocab.json
|-- audio2mesh.pt
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- film_net_fp16.pt
```

Tips :
The intermediate audio file will be generated and deleted,the raw video to pose video with audio and pose2video mp4 file will be located in the output directory of ComfyUI
the original uploaded mp4 video requires square size like 512x512, otherwise the result will be weird 
#### I've updated diffusers from 0.24.x to 0.26.2,so the diffusers/models/embeddings.py classname of PositionNet changed to GLIGENTextBoundingboxProjection and CaptionProjection changed to PixArtAlphaTextProjection,you should pay attention to it and modify the corresponding python files like src/models/transformer_2d.py if you installed the lower version of diffusers 

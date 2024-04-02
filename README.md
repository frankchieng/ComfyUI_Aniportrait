### reference audio combined with reference image and reference video
![audio2video](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/8172562e-daab-4e36-8a82-3648a11fa2c0)
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/51b1a7ab-854d-4f2d-8ba8-2fc0b92764dd" muted="false"></video>
    </td>
</tr>
</table>

### raw video to pose video with reference image
![pose2video](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/3467550f-3368-4a2c-9db8-884f4d50a492)
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/4f3bd91e-367a-435b-bedb-54c63df1d32f" muted="false"></video>
    </td>
</tr>
</table>



This is unofficial implementation of AniPortrait in ComfyUI custom_node,cuz i have routine jobs,so i will update this project when i have time
> [Aniportrait_pose2video.json](https://github.com/frankchieng/ComfyUI_Aniportrait/blob/main/Aniportrait_pose2video.json)

> [Audio driven](https://github.com/frankchieng/ComfyUI_Aniportrait/blob/main/audio2video_workflow.json)
> 
you should run
> git clone https://github.com/frankchieng/ComfyUI_Aniportrait.git
> 
then
> run pip install -r requirements.txt

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
`-- reference_unet.pth
```

Tips :
The intermediate audio file will be generated and deleted,the raw video to pose video with audio and pose2video mp4 file will be located in the output directory of ComfyUI
the original uploaded mp4 video requires square size like 512x512, otherwise the result will be weird 
#### I've updated diffusers from 0.24.x to 0.26.2,so the diffusers/models/embeddings.py classname of PositionNet changed to GLIGENTextBoundingboxProjection and CaptionProjection changed to PixArtAlphaTextProjection,you should pay attention to it and modify the corresponding python files like src/models/transformer_2d.py if you installed the lower version of diffusers 

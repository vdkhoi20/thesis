from fastapi import FastAPI, Request
import os
import torch
import json
import torch.nn.functional as F

from torchvision.utils import save_image,make_grid
from torchvision.io import read_image
from pytorch_lightning import seed_everything

import numpy as np
from diffusers import DDIMScheduler
from OIIctrl.diffuser_utils import OIICtrlPipeline
from OIIctrl.OIIctrl import OIISelfAttentionControlMaskExpand,OIISelfAttentionControlMask,OIISelfAttentionControlMaskExpandConcept,OIISelfAttentionControlMaskConcept
from OIIctrl.config import Config as cfg
from OIIctrl.OIIctrl_utils import load_image,expand_mask,regiter_attention_editor_diffusers,AttentionBase,get_ref_object_token_ids


model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)


import io
import base64
from PIL import Image

seed_everything(cfg.SEED)

model = OIICtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(cfg.DEVICE)
app = FastAPI()
def tensor_to_base64(tensor,orignal_size):
    image = tensor.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)    
    image = Image.fromarray(image)
    # resize to original_size
    image = image.resize(orignal_size)
    image.save("output.png")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_object_mask(image):
    
    if image.shape[0]>3: image=image[1]
    else: image=image[0]
    object_mask = image > 0.0  # Adjust the threshold if needed
    
    object_mask = object_mask.float()
   
    return object_mask
def process_image_mask(image_data,mask_data):

    image_bytes = base64.b64decode(image_data)
    image_bytes = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # print(image_bytes.size)
    image_bytes.save("input.png")
    image = torch.tensor(np.array(image_bytes)).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0  # [-1, 1]
    # print("SIZE-------------------------")
    # print(image.shape)
    orignal_size=(image.shape[-1],image.shape[-2])
    # print(original_size)

    image = F.interpolate(image, (512, 512))
    image = image.to(cfg.DEVICE)
    
    mask_bytes = base64.b64decode(mask_data)
    mask_bytes = Image.open(io.BytesIO(mask_bytes)).convert('RGB')
 
    mask_tensor = torch.tensor(np.array(mask_bytes)).permute(2, 0, 1).float()
    # breakpoint()
    mask_tensor=extract_object_mask(mask_tensor)
    mask_tensor = mask_tensor.to(cfg.DEVICE)
    
    return image,mask_tensor,orignal_size



def generate_add_item(source_image,source_mask,target_prompt):
    # target_mask=source_mask.detach().clone()
    target_mask=expand_mask(source_mask,0.1)
    # breakpoint()
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    editor = OIISelfAttentionControlMask(start_step=7, start_layer=17,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
                                        )
    

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_expand= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    k=cfg.k,
                    )
    
    
    return image_orginal_interpolate_intermediate_expand[2]


def generate_alter_background(source_image,source_mask,target_prompt):
    source_mask=1.0-source_mask
    target_mask=source_mask.detach().clone()
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    editor = OIISelfAttentionControlMask(start_step=7, start_layer=17,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
                                        )
    

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_expand= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    k=cfg.k,
                    )
    
    
    return image_orginal_interpolate_intermediate_expand[2]


def generate_change_pose_view(source_image,source_mask,target_prompt):
    # source_mask=1.0-source_mask
    target_mask=expand_mask(source_mask,0.1)
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    ref_token_ids_object=get_ref_object_token_ids(model,target_prompt,target_prompt)

    editor = OIISelfAttentionControlMaskExpand(start_step=3, start_layer=8,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
                                        ref_token_ids_object=ref_token_ids_object,
                                        thres_hold=cfg.THRES_HOLD,
                                        step_change_mask=cfg.STEP_CHANGE_MASK)
        

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

        
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_autorefining= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    return_intermediates=False,
                    k=cfg.k
                    )
    
    
    
    return image_orginal_interpolate_intermediate_autorefining[2]

def generate_replace_object_fixed(source_image,source_mask,target_prompt):
    # target_mask=source_mask.detach().clone()
    target_mask=expand_mask(source_mask,0.07)
    # breakpoint()
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    editor = OIISelfAttentionControlMask(start_step=7, start_layer=17,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
                                        )
    

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_expand= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    k=cfg.k,
                    )
    
    
    return image_orginal_interpolate_intermediate_expand[2]
def generate_replace_object_dynamic(source_image,source_mask,target_prompt):
    # source_mask=1.0-source_mask
    target_mask=expand_mask(source_mask,0.1)
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    ref_token_ids_object=get_ref_object_token_ids(model,target_prompt,target_prompt)

    editor = OIISelfAttentionControlMaskExpand(start_step=7, start_layer=17,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
                                        ref_token_ids_object=ref_token_ids_object,
                                        thres_hold=cfg.THRES_HOLD,
                                        step_change_mask=cfg.STEP_CHANGE_MASK)
        

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

        
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_autorefining= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    return_intermediates=False,
                    k=cfg.k
                    )
    
    
    
    return image_orginal_interpolate_intermediate_autorefining[2]

def generate_thematic_collection(source_image,source_mask,target_prompt):
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    target_mask=expand_mask(source_mask,0.1)
    
    ref_token_ids_object=get_ref_object_token_ids(model,target_prompt,target_prompt)
    
    editor = OIISelfAttentionControlMaskExpandConcept(start_step=7, start_layer=14,
                                            mask_s=source_mask,mask_t=target_mask,
                                            use_interpolate=cfg.USE_INTERPOLIATE,
                                            use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                            initial_alpha=cfg.INITIAL_ALPHA,
                                            total_steps=cfg.MAX_STEP,
                                            ref_token_ids_object=ref_token_ids_object,
                                            thres_hold=cfg.THRES_HOLD,
                                            step_change_mask=cfg.STEP_CHANGE_MASK)
        

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_autorefining= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    return_intermediates=False,
                    )
   
    
    
    
    

    
    return image_orginal_interpolate_intermediate_autorefining[2]

def generate_remove_object(source_image,source_mask,target_prompt):
    # source_mask=1.0-source_mask
    target_mask=torch.zeros_like(source_mask)

    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    editor = OIISelfAttentionControlMask(start_step=7, start_layer=17,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        )
    

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", ""]
    image_orginal_interpolate_intermediate_expand= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
        
                    )
    
    return image_orginal_interpolate_intermediate_expand[2]

def generate_thematic_collection_fixed(source_image,source_mask,target_prompt):
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    target_mask=expand_mask(source_mask,0.1)
    
    
    editor = OIISelfAttentionControlMaskConcept(start_step=7, start_layer=14,
                                            mask_s=source_mask,mask_t=target_mask,
                                            use_interpolate=cfg.USE_INTERPOLIATE,
                                            use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                            initial_alpha=cfg.INITIAL_ALPHA,
                                            total_steps=cfg.MAX_STEP,
                                            # ref_token_ids_object=ref_token_ids_object,
                                            # thres_hold=cfg.THRES_HOLD,
                                            # step_change_mask=cfg.STEP_CHANGE_MASK
                                            )
        

    regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

    
    prompts = ["","", target_prompt]
    image_orginal_interpolate_intermediate_expand= model(prompts,
                    latents=start_code,
                    ref_original=ref_original,
                    ref_intermediates=intermediates,
                    use_interpolate=cfg.USE_INTERPOLIATE,
                    use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
        
                    )
        
    
    
    
    

    
    return image_orginal_interpolate_intermediate_expand[2]

@app.post("/add_item")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']   
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        
        result=generate_add_item(image,mask,prompt)
        # print("result shapeee---------------------")
        # print(result.shape)
        
        image_bytes_code=tensor_to_base64(result,orignal_size)
        # convert result image pytorch tensor to bytes code and send back 
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}


@app.post("/alter_background")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt = data['prompt']
        

        image,mask,orignal_size=process_image_mask(image,mask)
        result = generate_alter_background(image,mask,prompt)
        
        
        image_bytes_code=tensor_to_base64(result,orignal_size)

        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}




@app.post("/change_pose_view")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        result=generate_change_pose_view(image,mask,prompt)
        
        image_bytes_code=tensor_to_base64(result,orignal_size)

        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}



@app.post("/replace_object_dynamic")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask.orignal_size=process_image_mask(image,mask)
        result=generate_replace_object_dynamic(image,mask,prompt)
        image_bytes_code=tensor_to_base64(result,orignal_size)
        
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}

@app.post("/replace_object_fixed")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        result=generate_replace_object_fixed(image,mask,prompt)
        image_bytes_code=tensor_to_base64(result,orignal_size)
        
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}


@app.post("/remove_object")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        result=generate_remove_object(image,mask,prompt)
        image_bytes_code=tensor_to_base64(result,orignal_size)
        
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}

@app.post("/thematic_collection")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        result=generate_thematic_collection(image,mask,prompt)
        image_bytes_code=tensor_to_base64(result,orignal_size)
        
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/thematic_collection_fixed")
async def read_data(request: Request):
    try:
        data = await request.json()  # Use await to get JSON data
        image = data['image']
        mask = data['mask']
        prompt=data['prompt']
        
        # print("prompt---------------------")
        # print(prompt)

        image,mask,orignal_size=process_image_mask(image,mask)
        result=generate_thematic_collection_fixed(image,mask,prompt)
        image_bytes_code=tensor_to_base64(result,orignal_size)
        
        return {"data": image_bytes_code}
    except Exception as e:
        return {"error": str(e)}
    

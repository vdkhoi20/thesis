import os
import torch
import json
import torch.nn.functional as F

from torchvision.utils import save_image,make_grid
from torchvision.io import read_image
from pytorch_lightning import seed_everything


from diffusers import DDIMScheduler
from OIIctrl.diffuser_utils import OIICtrlPipeline
from OIIctrl.OIIctrl import OIISelfAttentionControlMaskExpand,OIISelfAttentionControlMask
from OIIctrl.config import Config as cfg
from OIIctrl.OIIctrl_utils import load_image,expand_mask,regiter_attention_editor_diffusers,AttentionBase,get_ref_object_token_ids


model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = OIICtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(cfg.DEVICE)



root="my_datasets"

SOURCE_IMAGES_PATH = "./images"
SOURCE_MASKS_PATH="./masks"



def extract_object_mask(image):
    
    if image.shape[0]>3: image=image[1]
    else: image=image[0]
    object_mask = image > 0.0  # Adjust the threshold if needed
    
    object_mask = object_mask.float()
   
    return object_mask

with open(os.path.join(root,"data.json")) as fi:
    datas=json.load(fi)
# modify_results=['cat_3','cat','tree_1','bear3']
modify_results=["new_cat_3"]

task=f"specific_region"


for i,data in enumerate(datas):
    # if (i<100):
    #     continue
    if (data['retain_object']==True):
        # cfg.STEP_QUERY=3
        cfg.STEP_QUERY=3
        cfg.LAYER_QUERY=8
        continue
    else:
        
        cfg.STEP_QUERY=7
        cfg.LAYER_QUERY=17
        # continue
    # task_compare_FIOII
    source_image = load_image(os.path.join(root,SOURCE_IMAGES_PATH,data['img_name']), cfg.DEVICE)
    # source_prompt = image_path.split(".")[0]

    # if (source_prompt not in target_prompt_dict):
    #     continue
    # image_compose=[]
    # print(data)
    mask_file_name=data['img_name'].split(".")[0]
    if mask_file_name not in modify_results:
        continue
    # if data["target_text"]!="A photo of a cat wearing a hat.": continue
    if (data['alter_mask']!=None):
        mask_file_name=data['alter_mask']
        
    source_mask=extract_object_mask(read_image(os.path.join(root,SOURCE_MASKS_PATH,f"{mask_file_name}.png")).to(cfg.DEVICE)) # shape (H,W)
    
    target_prompt=data["target_text"]
    target_prompt_object=data["object_prompt"]

    if (data['object']=="background"):
        source_mask=1.0-source_mask
        # target_prompt_object="background"
    
    # cfg.SCALE_MASK=0.12
    target_mask=expand_mask(source_mask,cfg.SCALE_MASK)
    
    # source_mask=torch.zeros_like(target_mask)

    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(3, -1, -1, -1)
    ref_original=intermediates[0]
    
    
    
    # image_fixed = model([target_prompt],
    #                         latents=start_code[-1:],
    #                         num_inference_steps=cfg.MAX_STEP,
    #                         guidance_scale=cfg.GUIDANCE_SCALE)


    # if source_prompt=="messi" or source_prompt=="caribou":
    #     continue
        # print(source_prompt,target_prompt)
        # source_mask=read_image(os.path.join(SOURCE_MAKSS_PATH,f"{image_path}")).squeeze(0).to(cfg.DEVICE)/255 # shape (H,W)
        # source_mask=extract_object_mask(read_image(os.path.join(SOURCE_MAKSS_PATH,f"{image_path}")).to(cfg.DEVICE)) # shape (H,W)
        # tgt_mask = F.interpolate(source_mask.unsqueeze(0).unsqueeze(0), (512, 512))
        # tgt_mask = tgt_mask.repeat(1,3, 1, 1)
        # save_image(tgt_mask,f"{source_prompt}-{target_prompt}.png")
        # break
        
        # cfg.SCALE_MASK=0.1
        # target_mask=expand_mask(source_mask,cfg.SCALE_MASK)

        # editor = AttentionBase()

        # regiter_attention_editor_diffusers(model, editor)

        # start_code, intermediates = model.invert(source_image,
        #                                         prompt="",
        #                                         guidance_scale=cfg.GUIDANCE_SCALE,
        #                                         num_inference_steps=cfg.MAX_STEP,
        #                                         return_intermediates=True)

    

        # image_fixed = model([target_prompt],
        #                     latents=start_code[-1:],
        #                     num_inference_steps=cfg.MAX_STEP,
        #                     guidance_scale=cfg.GUIDANCE_SCALE)

        
        
        
        
    prompts = ["","", target_prompt]

        
        
    ref_token_ids_object=get_ref_object_token_ids(model,target_prompt,target_prompt_object)
    # breakpoint()
        # print("ref token ids object :", ref_token_ids_object)
    editor = OIISelfAttentionControlMaskExpand(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
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
    cross_attn_masks=editor.cross_attention_maps
    # print("length cross attention map",len(cross_attn_masks))
    # print(cross_attn_masks[0].shape)
    tgt_mask=editor.refining_mask
    tgt_mask = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0), (512, 512))
    tgt_mask = tgt_mask.repeat(1,3, 1, 1)
        
        # editor = OIISelfAttentionControlMaskExpand(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
        #                                     mask_s=source_mask,mask_t=target_mask,
        #                                     # use_interpolate=cfg.USE_INTERPOLIATE,
        #                                     # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                                     initial_alpha=cfg.INITIAL_ALPHA,
        #                                     total_steps=cfg.MAX_STEP,
        #                                     ref_token_ids_object=ref_token_ids_object,
        #                                     thres_hold=cfg.THRES_HOLD,
        #                                     step_change_mask=cfg.STEP_CHANGE_MASK)
        

        # regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

        
        # prompts = ["", target_prompt]
        # image_intermediate_autorefining= model(prompts,
        #                 latents=start_code,
        #                 # ref_original=ref_original,
        #                 ref_intermediates=intermediates,
        #                 # use_interpolate=cfg.USE_INTERPOLIATE,
        #                 # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                 guidance_scale=cfg.GUIDANCE_SCALE,
        #                 num_inference_steps=cfg.MAX_STEP,
        #                 )
        # tgt_mask_intermediate=editor.refining_mask
        # tgt_mask_intermediate = F.interpolate(tgt_mask_intermediate.unsqueeze(0).unsqueeze(0), (512, 512))
        # tgt_mask_intermediate = tgt_mask_intermediate.repeat(1,3, 1, 1)
        
        #----------------------------------------------------------------------------------------------------
    # target_mask=expand_mask(source_mask,cfg.SCALE_MASK)

    editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
                                        mask_s=source_mask,mask_t=target_mask,
                                        use_interpolate=cfg.USE_INTERPOLIATE,
                                        use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
                                        initial_alpha=cfg.INITIAL_ALPHA,
                                        total_steps=cfg.MAX_STEP,
                                        al=cfg.al,
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
                    k=cfg.k,
                
                    )
    
        
    
        


        # editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
        #                                     mask_s=source_mask,mask_t=target_mask,
        #                                     # use_interpolate=cfg.USE_INTERPOLIATE,
        #                                     # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                                     initial_alpha=cfg.INITIAL_ALPHA,
        #                                     total_steps=cfg.MAX_STEP,
        #                                     )
        # regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)
        # prompts = ["", target_prompt]
        # image_intermediate_expand= model(prompts,
        #                 latents=start_code,
        #                 # ref_original=ref_original,
        #                 ref_intermediates=intermediates,
        #                 # use_interpolate=cfg.USE_INTERPOLIATE,
        #                 # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                 guidance_scale=cfg.GUIDANCE_SCALE,
        #                 num_inference_steps=cfg.MAX_STEP,
            
        #                 )


        # cfg.SCALE_MASK=0
        # target_mask=expand_mask(source_mask,cfg.SCALE_MASK)

        # editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
        #                                     mask_s=source_mask,mask_t=target_mask,
        #                                     use_interpolate=cfg.USE_INTERPOLIATE,
        #                                     use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                                     initial_alpha=cfg.INITIAL_ALPHA,
        #                                     total_steps=cfg.MAX_STEP,
        #                                     # ref_token_ids_object=ref_token_ids_object,
        #                                     # thres_hold=cfg.THRES_HOLD,
        #                                     # step_change_mask=cfg.STEP_CHANGE_MASK
        #                                     )
        

        # regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)

        
        # prompts = ["","", target_prompt]
        # image_orginal_interpolate_intermediate_fixed= model(prompts,
        #                 latents=start_code,
        #                 ref_original=ref_original,
        #                 ref_intermediates=intermediates,
        #                 use_interpolate=cfg.USE_INTERPOLIATE,
        #                 use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                 guidance_scale=cfg.GUIDANCE_SCALE,
        #                 num_inference_steps=cfg.MAX_STEP,
            
        #                 )
        
        
    
        


        # editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
        #                                     mask_s=source_mask,mask_t=target_mask,
        #                                     # use_interpolate=cfg.USE_INTERPOLIATE,
        #                                     # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                                     initial_alpha=cfg.INITIAL_ALPHA,
        #                                     total_steps=cfg.MAX_STEP,
        #                                     )
        # regiter_attention_editor_diffusers(model, editor,use_foreground_mask=cfg.FOREGROUND_MASK)
        # prompts = ["", target_prompt]
        # image_intermediate_fixed= model(prompts,
        #                 latents=start_code,
        #                 # ref_original=ref_original,
        #                 ref_intermediates=intermediates,
        #                 # use_interpolate=cfg.USE_INTERPOLIATE,
        #                 # use_interpolate_inter=cfg.USE_INTERPOLIATE_INTER,
        #                 guidance_scale=cfg.GUIDANCE_SCALE,
        #                 num_inference_steps=cfg.MAX_STEP,
            
        #                 )


        # out_dir = os.path.join(root,img_cmp_path)
        
        
        # out_image = torch.cat([torch.cat([image_orginal_interpolate_intermediate_autorefining[0],
        #                             image_fixed[0],
        #                             image_orginal_interpolate_intermediate_autorefining[2],
        #                             image_intermediate_autorefining[1],
        #                             tgt_mask[0],
        #                             tgt_mask_intermediate[0],
        #                             ],dim=2),
        #                       image_orginal_interpolate_intermediate_expand[2],
        #                         image_intermediate_expand[1],
        #                         image_orginal_interpolate_intermediate_fixed[2],
        #                         image_intermediate_fixed[1],
        #                         ] , dim=1)
    
        # out_image=make_grid([
        #                     # image_orginal_interpolate_intermediate_autorefining[0],
        #                     # image_fixed[0],
        #                     image_orginal_interpolate_intermediate_autorefining[2],
        #                     # image_intermediate_autorefining[1],
                            
        #                     # tgt_mask_intermediate[0],
        #                     # image_orginal_interpolate_intermediate_expand[2],
        #                     # image_intermediate_expand[1],
        #                     image_orginal_interpolate_intermediate_fixed[2],
        #                     # tgt_mask[0],
        #                     # image_intermediate_fixed[1],
    #     #                     ],nrow=2)
    # image_compose=[]
    # for i,maskkk in enumerate(cross_attn_masks):
    #     # if (i%10!=0): continue
        
    #     tgt_mask=maskkk.clone().detach()
    #     min_value = tgt_mask.min()
    #     max_value = tgt_mask.max()
    #     tgt_mask = (tgt_mask - min_value) / (max_value-min_value)
        
    #     gr_eq=tgt_mask>=0.5
    #     lt=tgt_mask<0.5
    #     tgt_mask[gr_eq]=1
    #     tgt_mask[lt]=0
        
    #     tgt_mask = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0), (512, 512))
    #     tgt_mask = tgt_mask.repeat(1,3, 1, 1)
        
        
        
        # image_compose.append(tgt_mask[0])
    image_compose=[image_orginal_interpolate_intermediate_autorefining[1],
                #    image_fixed[0],
                image_orginal_interpolate_intermediate_autorefining[2],
                tgt_mask[0],   
                image_orginal_interpolate_intermediate_expand[2]]
    
    out_dir=f'{task}_3_8_layout'
    os.makedirs(out_dir, exist_ok=True)
    
    image_name=data['img_name'].split(".")[0]
    out_path=os.path.join(out_dir,f"{ image_name }_{data['target_text']}.png")
    
    # num=len(image_compose)//5
    out_images=make_grid(image_compose,nrow=4)              
    save_image(out_images, out_path)

    print("Syntheiszed images are saved in", out_path)


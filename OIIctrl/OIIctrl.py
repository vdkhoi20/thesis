import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .OIIctrl_utils import AttentionBase

from torchvision.utils import save_image


class OIISelfAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50,initial_alpha=0.23):
        """
        Original Interpolate Intermediate self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__(total_steps,initial_alpha)
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        
        if kwargs.get("is_mask") is not None or is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        
        return out
class OIISelfAttentionControlMask(OIISelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 use_interpolate=False,
                 use_interpolate_inter=False,
                 initial_alpha=0.23,al=0.4,
                 mask_s=None, mask_t=None, mask_save_dir=None):
        """
        Maske-guided Original Interpolate Intermediate to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps,initial_alpha)

        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask

        self.use_interpolate=use_interpolate
        self.use_interpolate_inter=use_interpolate_inter
        self.al=al
        print("Using mask-guided Original Interpolate Intermediate Ctrl Always Query BackGround")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_s.png"))
            if self.mask_t is not None:
                save_image(self.mask_t.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_t.png"))

    def getValueInterpoliate(self,v,ref_v,num_heads):
        # ref_v = rearrange(ref_v, "h n d -> (h n) d", h=num_heads)
        # v=self.alpha*ref_v+(1-self.alpha)*v
        v=self.al*ref_v+(1-self.al)*v
        return v
    def attn_batch(self, q, k, v, num_heads,background=True,ref_v=None, **kwargs):
        H = W = int(np.sqrt(q.shape[1]))
        
        q = rearrange(q, "(b h) n d -> b h n d", h=num_heads)
        # k = rearrange(k, "h n d -> (h n) d", h=num_heads)
        # v = rearrange(v, "h n d -> (h n) d", h=num_heads)

        sim = torch.einsum("b h i d,h j d -> b h i j", q, k) * kwargs.get("scale")
        
        # sim=rearrange(sim,"b j (h n) -> b j h n",h=num_heads)
        mask_sr = self.mask_s.unsqueeze(0).unsqueeze(0)
        mask_sr = F.interpolate(mask_sr, (H, W)).flatten(0).unsqueeze(0)
        mask_sr = mask_sr.flatten().to(sim.dtype)
        # background
        if background:
            # v=self.getValueInterpoliate(v,ref_v,num_heads)
            sim = sim + mask_sr.masked_fill(mask_sr == 1, torch.finfo(sim.dtype).min)
        # object
        else:
            v=self.getValueInterpoliate(v,ref_v,num_heads)
            mask_sr = self.getMask().unsqueeze(0).unsqueeze(0)
            # mask_sr = self.mask_s.clone().unsqueeze(0).unsqueeze(0)
            mask_sr = F.interpolate(mask_sr, (H, W)).flatten(0).unsqueeze(0)
            mask_sr = mask_sr.flatten().to(sim.dtype)
            
            
            # mask_sr = self.getMask().unsqueeze(0).unsqueeze(0)
            
            
            sim = sim + mask_sr.masked_fill(mask_sr == 0, torch.finfo(sim.dtype).min).masked_fill(mask_sr == 1,0)
        
        # sim=rearrange(sim," b j h n -> b j (h n)",h=num_heads)
        attn = sim.softmax(-1)

        # breakpoint()

        out = torch.einsum("b h i j, h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=num_heads)
        return out
    def getMask(self):
        return self.mask_t.clone()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """

        H = W = int(np.sqrt(q.shape[1]))
        out_self_attn=super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, is_mask=True,**kwargs)
        if is_cross:
            mask_fg_cross = self.getMask().unsqueeze(0).unsqueeze(0)
            mask_fg_cross = F.interpolate(mask_fg_cross, (H, W)).flatten(0).unsqueeze(0)
            mask_fg_cross = mask_fg_cross.flatten()
            
            # breakpoint()

            q_object=q[-num_heads:,mask_fg_cross==1]
            k_object=k[-num_heads:]
            v_object=v[-num_heads:]
            
            sim_object = torch.einsum('b i d, b j d -> b i j', q_object, k_object) *kwargs.get("scale") # (b h) no nt
            attn_object= sim_object.softmax(dim=-1) # 8*128*77
            out_object = torch.einsum('b i j, b j d -> b i d', attn_object, v_object) 

            
            q_background=q[-num_heads:,mask_fg_cross==0]
            k_background=k[:num_heads]
            v_background=v[:num_heads]
            
            
            sim_background = torch.einsum('b i d, b j d -> b i j', q_background, k_background) *kwargs.get("scale") # (b h) no nt
            attn_background= sim_background.softmax(dim=-1) # 8*128*77
            out_background = torch.einsum('b i j, b j d -> b i d', attn_background, v_background) 
            
            # v[-num_heads:,:len_rm+1,:]=v[:num_heads:,:len_rm+1:]
            out = torch.einsum('b i j, b j d -> b i d', attn, v) 
            
            # breakpoint()
            # q shape (32,4096,40)
            # k shape (32,77,40)
            # v shape (32,77,40) 
            
            out[-num_heads:,mask_fg_cross==1]=out_object
            out[-num_heads:,mask_fg_cross==0]=out_background

            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            
            return out

    
        out_source ,out_intermediate,out_u_target,out_c_target=out_self_attn.chunk(4)
        
        
        out_target_bg_u,out_target_bg_c = self.attn_batch(q[-2*num_heads:], k[-3*num_heads:-2*num_heads], v[-3*num_heads:-2*num_heads], num_heads,background=True,ref_v=v[:num_heads], **kwargs).chunk(2)
        out_target_fg_u,out_target_fg_c = self.attn_batch(q[-2*num_heads:], k[-3*num_heads:-2*num_heads], v[-3*num_heads:-2*num_heads], num_heads,background=False,ref_v=v[:num_heads], **kwargs).chunk(2)
        

        mask=self.getMask()
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W))
        mask = mask.reshape(-1, 1)  # (hw, 1)
        
        # smooth mask
        mask = mask.clamp(0.01, 0.99).to("cuda")

        if kwargs.get("foreground_mask")  and  self.cur_step  in self.step_idx and self.cur_att_layer // 2  in self.layer_idx:
            out_u_target = out_target_fg_u * mask + out_target_bg_u * (1 - mask)
            out_c_target = out_target_fg_c * mask + out_target_bg_c * (1 - mask)
            
        else:
            out_u_target = out_u_target * mask  + out_target_bg_u * (1 - mask)
            out_c_target = out_c_target * mask  + out_target_bg_c * (1 - mask)


   
        out = torch.cat([out_source,out_intermediate, out_u_target,out_c_target], dim=0)

        return out
class OIISelfAttentionControlMaskExpand(OIISelfAttentionControlMask):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 use_interpolate=False,
                 use_interpolate_inter=False,
                 initial_alpha=0.23,al=0.4,
                 mask_s=None, mask_t=None, mask_save_dir=None,
                 thres_hold=0.25,
                 ref_token_ids_object=[1],
                 step_change_mask=5):
        """
        Maske-guided Original Interpolate Intermediate to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps,
                         use_interpolate,
                         use_interpolate_inter,
                         initial_alpha,al,
                         mask_s,mask_t,mask_save_dir)

        self.step_change_mask=step_change_mask
        self.thres_hold=thres_hold
        self.refining_masks=[]
        self.cross_attention_maps=[]
        self.refining_mask=None
        self.ref_token_ids_object=ref_token_ids_object
        self.num_cross_attention=0
    def aggregate_cross_attn_map(self):
        attns=torch.stack(self.cross_attention_maps,dim=0)
        attns= attns.sum(dim=0)/self.num_cross_attention

        min_value = attns.min()
        max_value = attns.max()
        attns = (attns - min_value) / (max_value-min_value)

        gr_eq=attns>=self.thres_hold
        lt=attns<self.thres_hold
        attns[gr_eq]=1
        attns[lt]=0

        return attns
    def add_attn_map(self,num_heads,attn):
        
        attn_ra=attn[-num_heads:]
        size=int(attn_ra.shape[1]**0.5)
        attn_ra=attn_ra[...,self.ref_token_ids_object]
        attn_ra=rearrange(attn_ra, "he (w h) d -> he d w h",w=size)
        attn_ra=F.interpolate(attn_ra, size=(64, 64), mode='nearest')
        attn_ra=rearrange(attn_ra, "he d w h -> (he d) w h")
        attn_ra=attn_ra.sum(dim=0)/attn_ra.shape[0]/torch.max(attn_ra) # W,H

 
        self.cross_attention_maps.append(attn_ra)
    def getMask(self):
        if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2): 
            return self.refining_mask.clone()
        return super().getMask()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            #auto refining mask
            self.num_cross_attention+=1
            if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2):
                self.add_attn_map(num_heads,attn)
                if (self.num_cross_attention%(self.num_att_layers//2)==0):
                    self.refining_mask=self.aggregate_cross_attn_map()
                    self.refining_masks.append(self.refining_mask)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)





class OIISelfAttentionControlMaskConcept(OIISelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 use_interpolate=False,
                 use_interpolate_inter=False,
                 initial_alpha=0.23,
                 mask_s=None, mask_t=None, mask_save_dir=None):
        """
        Maske-guided Original Interpolate Intermediate to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps,initial_alpha)

        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask

        self.use_interpolate=use_interpolate
        self.use_interpolate_inter=use_interpolate_inter

        print("Using mask-guided Original Interpolate Intermediate Ctrl Always Query BackGround")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_s.png"))
            if self.mask_t is not None:
                save_image(self.mask_t.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_t.png"))

    def getValueInterpoliate(self,v,ref_v,num_heads):
        ref_v = rearrange(ref_v, "(b h) n d -> h (b n) d", h=num_heads)
        if len(v) == 2 * len(ref_v):
            ref_v=torch.cat([ref_v]*2)
        v=self.alpha*ref_v+(1-self.alpha)*v
        return v
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,ref_v=None, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten().to(sim.dtype)
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min).masked_fill(mask == 1,0)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)

        if self.use_interpolate and ref_v!=None:
            v=self.getValueInterpoliate(v,ref_v,num_heads)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out
    def getMask(self):
        return self.mask_t.clone()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """

        H = W = int(np.sqrt(q.shape[1]))
        out_self_attn=super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, is_mask=True,**kwargs)
        if is_cross:
            mask_fg_cross = self.getMask().unsqueeze(0).unsqueeze(0)
            mask_fg_cross = F.interpolate(mask_fg_cross, (H, W)).flatten(0).unsqueeze(0)
            mask_fg_cross = mask_fg_cross.flatten()
            
            # 8 8 8 8
            # qu, qc = q.chunk(2)
            # ku, kc = k.chunk(2)
            # vu, vc = v.chunk(2)
            
            # text_embeddings cat uncodition shape torch.Size([4, 77, 768])
            # torch.Size([32, 4096, 40]) torch.Size([32, 77, 40]) torch.Size([32, 77, 40])
            # print(q.shape,k.shape,v.shape)
            q_object=q[-num_heads:,mask_fg_cross==1,:]
            k_object=k[-num_heads:,:,:]
            v_object=v[-num_heads:,:,:]
            
            # q_object=q[:,mask_fg_cross==1,:]
            # k_object[:,:6,:]*=3
            sim_object = torch.einsum('b i d, b j d -> b i j', q_object, k_object) *kwargs.get("scale") # (b h) no nt
            # sim_object = torch.einsum('b i d, b j d -> b i j', q, k) *kwargs.get("scale") # (b h) no nt
            # sim_object[:,:,:1:6]*=2
            attn_object= sim_object.softmax(dim=-1) # 8*128*77
            # v_object[:,:5,:]*=4
            out_object = torch.einsum('b i j, b j d -> b i d', attn_object, v_object) 

            
            q_background=q[-num_heads:,mask_fg_cross==0,:]
            k_background=k[:num_heads:,:,:]
            v_background=v[:num_heads:,:,:]
            
            
            sim_background = torch.einsum('b i d, b j d -> b i j', q_background, k_background) *kwargs.get("scale") # (b h) no nt
            # sim_object = torch.einsum('b i d, b j d -> b i j', q, k) *kwargs.get("scale") # (b h) no nt
            attn_background= sim_background.softmax(dim=-1) # 8*128*77
            out_background = torch.einsum('b i j, b j d -> b i d', attn_background, v_background) 
            
            out = torch.einsum('b i j, b j d -> b i d', attn, v) 
            out[-num_heads:,mask_fg_cross==1,:]=out_object
            out[-num_heads:,mask_fg_cross==0,:]=out_background

            # q[:,mask_fg_cross==1,:]=out
            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            
            return out

        H = W = int(np.sqrt(q.shape[1]))
        
        # qu, qc = q.chunk(2)
        # ku, kc = k.chunk(2)
        # vu, vc = v.chunk(2)
        # attnu, attnc = attn.chunk(2)

        if (self.use_interpolate_inter):
            out_source ,out_intermediate,out_u_target,out_c_target=out_self_attn.chunk(6)
            out_u_target_mask = self.attn_batch(q[-2*num_heads:-num_heads], k[:num_heads], v[num_heads:2*num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, is_mask_attn=True,ref_v=v[:num_heads], **kwargs)
            out_c_target_mask = self.attn_batch(q[-num_heads:], k[:num_heads], v[num_heads:2*num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, is_mask_attn=True,ref_v=v[:num_heads], **kwargs)
        else:
            out_intermediate,out_u_target,out_c_target=out_self_attn.chunk(4)
            out_u_target_mask = self.attn_batch(q[-2*num_heads:-num_heads], k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, is_mask_attn=True,ref_v=None, **kwargs)
            out_c_target_mask = self.attn_batch(q[-num_heads:], k[:num_heads], v[:num_heads], sim[:num_heads], attn, is_cross, place_in_unet, num_heads, is_mask_attn=True,ref_v=None, **kwargs)
        

        out_u_target_fg, out_u_target_bg = out_u_target_mask.chunk(2, 0)
        out_c_target_fg, out_c_target_bg = out_c_target_mask.chunk(2, 0)    

        if (self.mask_t is None):
            if kwargs.get("foreground_mask")  and  self.cur_step  in self.step_idx and self.cur_att_layer // 2  in self.layer_idx:
                out_u_target = out_u_target_fg  + out_u_target_bg 
                out_c_target = out_c_target_fg  + out_c_target_bg 
                
            else:
                out_u_target = out_u_target   + out_u_target_bg 
                out_c_target = out_c_target   + out_c_target_bg 
            # pass
        else:
            mask=self.getMask()
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            
            # smooth mask
            mask = mask.clamp(0.01, 0.99).to("cuda")

            if kwargs.get("foreground_mask")  and  self.cur_step  in self.step_idx and self.cur_att_layer // 2  in self.layer_idx:
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)
                
            else:
                out_u_target = out_u_target * mask  + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target * mask  + out_c_target_bg * (1 - mask)


        if (not self.use_interpolate_inter):
            out = torch.cat([out_intermediate, out_u_target ,out_c_target], dim=0)
        else:
            out = torch.cat([out_source,out_intermediate, out_u_target,out_c_target], dim=0)

        return out
class OIISelfAttentionControlMaskExpandConcept(OIISelfAttentionControlMaskConcept):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 use_interpolate=False,
                 use_interpolate_inter=False,
                 initial_alpha=0.23,
                 mask_s=None, mask_t=None, mask_save_dir=None,
                 thres_hold=0.25,
                 ref_token_ids_object=[1],
                 step_change_mask=5):
        """
        Maske-guided Original Interpolate Intermediate to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps,
                         use_interpolate,
                         use_interpolate_inter,
                         initial_alpha,
                         mask_s,mask_t,mask_save_dir)

        self.step_change_mask=step_change_mask
        self.thres_hold=thres_hold
        self.refining_masks=[]
        self.cross_attention_maps=[]
        self.refining_mask=None
        self.ref_token_ids_object=ref_token_ids_object
        self.num_cross_attention=0
    def aggregate_cross_attn_map(self):
        attns=torch.stack(self.cross_attention_maps,dim=0)
        attns= attns.sum(dim=0)/self.num_cross_attention

        min_value = attns.min()
        max_value = attns.max()
        attns = (attns - min_value) / (max_value-min_value)

        gr_eq=attns>=self.thres_hold
        lt=attns<self.thres_hold
        attns[gr_eq]=1
        attns[lt]=0

        return attns
    def add_attn_map(self,num_heads,attn):
        
        attn_ra=attn[-num_heads:]
        size=int(attn_ra.shape[1]**0.5)
        attn_ra=attn_ra[...,self.ref_token_ids_object]
        attn_ra=rearrange(attn_ra, "he (w h) d -> he d w h",w=size)
        attn_ra=F.interpolate(attn_ra, size=(64, 64), mode='nearest')
        attn_ra=rearrange(attn_ra, "he d w h -> (he d) w h")
        attn_ra=attn_ra.sum(dim=0)/attn_ra.shape[0]/torch.max(attn_ra) # W,H

 
        self.cross_attention_maps.append(attn_ra)
    def getMask(self):
        if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2): 
            return self.refining_mask.clone()
        return super().getMask()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            #auto refining mask
            self.num_cross_attention+=1
            if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2):
                self.add_attn_map(num_heads,attn)
                if (self.num_cross_attention%(self.num_att_layers//2)==0):
                    self.refining_mask=self.aggregate_cross_attn_map()
                    self.refining_masks.append(self.refining_mask)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

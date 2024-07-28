
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.io import read_image
import torch.nn.functional as F
import cv2 
from rembg import remove
from PIL import Image
import numpy as np
import io
from torchvision.utils import save_image,make_grid


class AttentionBase:
    def __init__(self,max_step=50,initial_alpha=0.23):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        self.max_step=max_step
        self.initial_alpha=initial_alpha
        self.initial_ld=0.15

        
        
        
        self.ld=self.initial_ld
        self.alpha=self.initial_alpha

    def reset(self):
        self.alpha=self.initial_alpha
        self.ld=self.initial_ld
        self.cur_att_layer = 0
        self.cur_step = 0

    def after_step(self):
        self.cur_att_layer = 0
        self.cur_step += 1

        self.alpha+=1/self.max_step
        self.alpha=min(self.alpha,1.0)

        self.ld+=2.3/self.max_step
        self.ld=min(self.ld,1.0)
 

        self.cur_step%=self.max_step
        if self.cur_step == 0:
            self.reset()
    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out


def regiter_attention_editor_diffusers(model, editor: AttentionBase,use_foreground_mask=True):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale,foreground_mask=use_foreground_mask)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image
def post_process(image):
    brightness_reduction = 0.05  #reduce 5%
    brightened_image = torch.clamp(image - brightness_reduction, 0, 1)
    return brightened_image
def expand_mask(mask,scale=0.15):
    
    object_size = torch.sum(mask)
    kernel_size = int(torch.sqrt(object_size).item()*scale)
    if (kernel_size==0): return mask 
    source_mask_tensor = torch.tensor(mask.clone().detach(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    dilation = torch.ones(1, 1, kernel_size, kernel_size).to(source_mask_tensor.device) 
    
    expanded_mask_tensor = F.conv2d(source_mask_tensor, dilation, padding=kernel_size//2)
    expanded_mask_tensor = torch.where(expanded_mask_tensor > 0, torch.tensor(1.0).to(source_mask_tensor.device), torch.tensor(0.0).to(source_mask_tensor.device))
    expanded_mask = expanded_mask_tensor.squeeze().byte()

    return expanded_mask
def get_ref_object_token_ids(model,sentence,object):
    prompt=[sentence,object]

    ids = model.tokenizer(
            prompt, # todo
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
    # print(ids['input_ids'],sentence,object)
    object_token_ids=ids['input_ids'][1]
    padding_token_id=object_token_ids[-1].item()
    ref_tokens_object=[]
    for token in object_token_ids[1:]:

        if token==padding_token_id:
            break
        ref_tokens_object.append(token.item())
    ref_tokens_object=torch.tensor(ref_tokens_object)

    sentence_token_ids=ids['input_ids'][0]

    for first_id in range(len(sentence_token_ids) - len(ref_tokens_object) + 1):
        if torch.equal(sentence_token_ids[first_id:first_id+len(ref_tokens_object)], ref_tokens_object):
            break
    assert first_id<len(sentence_token_ids)-1,"token object must is a sequence of sentence"
    return list(range(first_id,first_id+len(ref_tokens_object)))
def save_image_pil(tensor, filename):
    img = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img_pil = Image.fromarray(img)
    img_pil.save(filename)
def remove_background(image):
    # image is numpy image
    # print(image.shape) # 462*474*3
    image_bytes = cv2.imencode(".png", image)[1].tobytes()
    output_data = remove(image_bytes)
    image_PIL=Image.open(io.BytesIO(output_data))
    image_PIL.save("cc2.png")
    result_array = np.array(image_PIL)

    image_torch =torch.from_numpy(result_array).permute( 2, 0, 1).to(torch.uint8)
    # print(image_torch[0])# 4*462*474
    # print(torch.max(image_torch[0]),torch.min(image_torch[0]))
    # save_image_pil(image_torch,f"cc2.png")
    # print(image_torch)
    object_mask = image_torch[0] > 0.0  # Adjust the threshold if needed 
    object_mask = object_mask.float()
      
    return object_mask

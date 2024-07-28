import torch
a=torch.rand(1,3,4,5)


orignal_size=(a.shape[-2],a.shape[-1])
print(type(orignal_size),orignal_size)
# from PIL import Image
# import numpy as np
# img=Image.open("test.png")

# cc=np.array(img)
# print(img.resize(tuple(a.shape[2:]))
# print(img.size,cc.shape)
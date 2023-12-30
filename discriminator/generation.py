from min_dalle import MinDalle
from IPython.display import display, update_display
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cuda',
    is_mega=True, 
    is_reusable=True
)


# temp = [0.1,0.5,1,2,3,4,5,6,7,8,9,10]
# top = [16,32,64,128,256,512,1024]
# supercondition = [8,16,32,64,128,256,512]

# # 여러장 한번에 확인

# for i in temp :
#     for j in top:
#         for k in supercondition:
#             image = model.generate_image(
#                 text='a surrealism melting clock drawn by Salvador Dali',
#                 seed=42,
#                 grid_size=5,
#                 is_seamless=False,
#                 temperature=i,
#                 top_k=j,
#                 supercondition_factor=k,
#                 is_verbose=False
#             )

#             output_path = '/home/tjddms9376/cv/result_figure/'
#             image.save(os.path.join(output_path,f'temp_{i}_top_{j}_spc_{k}.png'))

# 한장씩 저장 (generate_images로 여러장 생성 가능)
# images = model.generate_image(
#     text='Draw the paintings by Vincent Van Gogh so that you can see as many different objects as possible in each image.',
#     seed=-1,
#     grid_size=5,
#     is_seamless=False,
#     temperature=4,
#     top_k=1024,
#     supercondition_factor=16,
#     is_verbose=False
# )
# output_path = '/home/tjddms9376/cv/'
# images.save(os.path.join(output_path,'test.png'))

# 여러장 생성 시.

output_path = '/home/tjddms9376/cv/test/'
total_image = []

for s in tqdm(range(1)):
    images = model.generate_images(
        text='Draw the paintings by Vincent Van Gogh',
        seed=-1,
        grid_size=5,
        is_seamless=False,
        temperature=5,
        top_k=1024,
        supercondition_factor=16,
        is_verbose=False
    )
    for i in range(len(images)):
        image = images[i].to('cpu').numpy()
        image = Image.fromarray(image.astype(np.uint8))
        total_image.append(image)

for k in range(len(total_image)):
    total_image[k].save(os.path.join(output_path,f'{k}.png'))

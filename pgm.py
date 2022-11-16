import numpy as np
import math
import tensornetwork as tn
from albumentations.augmentations.crops.transforms import CenterCrop
import logging

def create_node(matrix, tensor_base):
  
  if len(matrix.shape) > 2:
    logging.error(f"create_node error: len(matrix.shape) ({len(matrix.shape)}) is more than {2}")
    return None, None
  
  pix_am = matrix.shape[0] * matrix.shape[1]
  N = math.log(pix_am, tensor_base)
  if not N.is_integer():
    logging.error(f"create_node error: the tensor rank {N} is not an integer")
    return None, None

  return tn.Node(np.reshape(matrix, tuple([tensor_base] * int(N)))), int(N) 

def crop_pgm_image(im_pgm, width=512, height=512):
  resizer = CenterCrop(width=width, height=height)
  im_pgm = resizer(image=im_pgm)["image"]
  
  return im_pgm 

def create_node_from_pgm(im_pgm, tensor_base, width=512, height=512):
  
  if im_pgm.shape[1] != width or im_pgm.shape[0] != height: 
    im_pgm = crop_pgm_image(im_pgm, width=width, height=height)
  
  return create_node(im_pgm, tensor_base)
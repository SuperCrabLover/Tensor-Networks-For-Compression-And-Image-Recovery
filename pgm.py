import numpy as np
import math
import tensornetwork as tn
from albumentations.augmentations.crops.transforms import CenterCrop

def create_tensor(matrix, tensor_base):
  '''
  INPUT:
  matrix     -- 2d ndarray -- image to tensorize
  tensor_base --    int     -- the dimension of each mode for a tensor
  
  OUTPUT:
  tensor     -- Nd ndarray -- an image in tensor represenation
  N          --    int     -- amount of modes in output tensor (N = log_{tensor_base}(matrix.shape[0] * matrix.shape[1]))
  '''
  
  pix_am = matrix.shape[0] * matrix.shape[1]
  N = math.log(pix_am, tensor_base)
  if not N.is_integer():
    raise Exception(f"create_node error: the tensor rank {N} is not an integer")
  
  N = int(N)
  return np.reshape(matrix, tuple([tensor_base] * N)), N

def create_node(matrix, tensor_base):
  '''
  INPUT:
  matrix     -- 2d ndarray -- image to tensorize
  tensor_base --    int     -- the dimension of each mode for a tensor
  
  OUTPUT:
  Node       -- Tensor Network Node -- an image in TN Node represenation
  N          --         int         -- amount of modes in output tensor (N = log_{tensor_base}(matrix.shape[0] * matrix.shape[1]))
  '''
  
  if len(matrix.shape) > 2:
    raise Exception(f"create_node error: len(matrix.shape) ({len(matrix.shape)}) is more than {2}")
  
  tensor, N = create_tensor(matrix, tensor_base)
  
  return tn.Node(tensor), N
  
  

def crop_pgm_image(im_pgm, width=512, height=512):
  '''
  INPUT:
  im_pgm -- 2d ndarray -- input image to crop
  width  --     int    -- the width of output image
  height --     int    -- the height of output image
  
  OUTPUT:
  im_pgm -- 2d ndarray -- cropped image 
  '''
  
  resizer = CenterCrop(width=width, height=height)
  im_pgm = resizer(image=im_pgm)["image"]
  
  return im_pgm 

def create_node_from_pgm(im_pgm, tensor_base, width=512, height=512):
  '''
  INPUT:
  im_pgm -- 2d ndarray -- input image to crop
  width  --     int    -- the width of output image
  height --     int    -- the height of output image
  tensor_base --    int     -- the dimension of each mode for a tensor
  
  OUTPUT:
  Node       -- Tensor Network Node -- an image in TN Node represenation
  N          --         int         -- amount of modes in output tensor (N = log_{tensor_base}(matrix.shape[0] * matrix.shape[1]))
  '''
  
  if im_pgm.shape[1] != width or im_pgm.shape[0] != height: 
    im_pgm = crop_pgm_image(im_pgm, width=width, height=height)
  
  return create_node(im_pgm, tensor_base)
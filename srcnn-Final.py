#Importing the libraries
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import misc
from scipy import ndimage
import skimage.io 
import skimage.metrics
from skimage.metrics import peak_signal_noise_ratio 
import imageio
from PIL import Image
from torchvision.utils import save_image
import torchvision
import cv2



def imread(path, is_grayscale=True):
    """
    Read image from the giving path.
    Default value is gray-scale, and image is read by YCbCr format as the paper.
    """
    if is_grayscale:
        return misc.imread(path, flatten=True, mode='YCbCr').astype(np.float32)
    else:
        return misc.imread(path, mode='YCbCr').astype(np.float32)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    label_ = label_ / 255.

    input_ = ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


"""Define the model weights and biases 
"""
## ------ Add your code here: set the weight of three conv layers
# replace 'None' with your hyper parameter numbers
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5

# OBSERVATIONS:
'''           Using different combinations of padding value effects the PSNR value,
              For all 3 Convolution network the respected padding values give the following PSNR values:
              padding - [(1,1),(1,1),(4,4)]   PSNR: 21.11
              padding - [(1,1),(4,4),(1,1)]   PSNR: 21.45
              padding - [(4,4),(1,1),(1,1)]   PSNR: 27.94  (MAX out of All)

              This observation made me think that having more padding on the first layer must give us a higher PSNR
              which means, higher resolution. But this statment was contradicted when i used the padding sizes as follows:
              
              padding - [(5,5),(1,1)]   PSNR: 27.70  (A slight decline, which then followed by the observation that we MIGHT not need
              to add any padding at the bottom layer) 
              
              This statemnt was also contradicted by the following padding values 
              padding - [(6,6)].   PSNR: 27.61

              Having higher padding value CAN increase the PSNR (but not always)
'''
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9,9), padding=(4,4), bias = True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), padding=(1,1), bias = True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(5,5), padding=(1,1), bias = True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out


"""Load the pre-trained model file
"""
model = SRCNN()
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()

"""Read the test image
"""
LR_image, HR_image = preprocess('./image/butterfly_GT.bmp')
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(LR_image, axis=0), axis=0)
input_ = torch.from_numpy(input_)

#Checking the shape of the input tensor
print("The Shape of the input image-tensor is : " , input_.shape , "which is has " , input_.ndim , "Dimensions") 




"""Run the model and get the SR image
"""
with torch.no_grad():
    output_ = model(input_)

    '''' 
        Was unable to use scipy.misc becaus of some version issues so used the 
        alternative method instead.
    '''

    tensor = output_
    img1 = tensor[0]

    ##Checking the shape of the output tensor
    print("The Shape of the output image-tensor is : " , tensor.shape , "which is has " , tensor.ndim , "Dimensions") 

    save_image(img1, 'img1.bmp')   # Saving the image 
    
    '''' 
        Was unable to use skimage becaus of some version issues so used the 
        alternative method instead. Also there was another error that i would like to discuss 
        with you. 
    '''
    #Calculating PSNR
    def psnr(label, outputs, max_val=1.):
      label = label.cpu().detach().numpy()
      outputs = outputs.cpu().detach().numpy()
      img_diff = outputs - label
      rmse = math.sqrt(np.mean((img_diff) ** 2))
      if rmse == 0:
        return 100
      else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
    PSNR = psnr(input_ , output_)

    #Printing the PSNR value 
    print("The PSNR value is: " '%.2f' % PSNR )
    
    #skimage.metrics.peak_signal_noise_ratio(input_, tensor)
##------ Add your code here: save the LR and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.metrics.peak_signal_noise_ratio()'





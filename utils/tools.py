import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from torchvision import transforms
from torch.distributions import Categorical


#------------------------------------
def simple_plot(fnames, grades, path_folder, size=(3,3), fs=12):
    fig = plt.figure(figsize=size)
    n = len(fnames)
    j = 0
    
    for i in range(n):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = plt.imread(os.path.join(path_folder, fnames[i]))
        ax.imshow(img)
        ax.set_title(f"{fnames[i].split('.')[0], grades[i]}", loc="center", fontsize=fs)
        ax.axis('off')
    plt.show()

#------------------------------------
def load_image(cfg, path, img_name, xchest=True):
    '''
    return the corresponding image (tensor and numpy): => (batch_size, C, H, W).

        Parameters:
            - path (str): image path location
        
        Returns:
            - PIL normalize (in [0, 1]) with shape (b, C,H,W)  
            - np image unormalize image
    '''
    pil_img = Image.open(os.path.join(path, img_name)) 

    if xchest:
        pil_img = pil_img.convert('RGB')
    
    normalization = [
        transforms.Resize((cfg.data.input_size)), 
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)]
    
    test_preprocess_norm = transforms.Compose(normalization)
    
    ts_img = torch.unsqueeze(test_preprocess_norm(pil_img), dim=0)   
    pil_img = pil_img.resize((cfg.data.input_size, cfg.data.input_size))
    np_img = np.array(pil_img)                                
    
    return ts_img, np_img

#------------------------------------
def get_tte_prediction(network, img, h=16, w=16):
    avgpool = nn.AvgPool2d(kernel_size=(h, w), stride=(1,1), padding=0) 

    activation = network(img)
    acti = activation.squeeze().data.cpu().numpy()
        
    out = avgpool(activation)
    
    prediction = out.view(out.shape[0], -1)
    prediction = prediction.data.cpu()
    
    dist = Categorical(logits=prediction)
    pred, yhat = torch.topk(dist.probs, 1)
    pred = round(pred.item(), 4)
    
    return acti, pred, yhat.item()

#------------------------------------
def get_overlay_img(img, activation, alpha = 0.6):
    '''
        alpha: Transparency factor
        only normalize for conv evidence map
    '''

    # Normalize the heatmap data to the range [0, 1]
    heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation))
    
    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend the image and heatmap
    overlay_image = cv2.addWeighted(img, 1 - alpha, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), alpha, 0)
    return overlay_image

#------------------------------------
def plot_img_heat_att(df, imgs, dic, size, fs=10, l=1): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fname = dic[0]
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(imgs)
    j = 0
    
    for i, img in enumerate(imgs):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = ax.imshow(img, cmap='viridis')
        if i > 0: 
            ax.set_title(dic[i], loc="center", fontsize=fs)

        tmp = df[df.patientId==fname]
        for _, row in tmp.iterrows():
            rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=l, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
        ax.axis('off')
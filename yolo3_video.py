from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
import os 
import os.path as osp

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


def draw_save(videofile):
    ##改
    save_file='test_video'

    confidence=0.5
    nms_thesh=0.4
    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    classes = load_classes('data/coco.names')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("yolov3.weights")
    model.net_info["height"] = "416"
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    try:
        imlist = [osp.join(osp.realpath('.'), videofile, img) for img in os.listdir(videofile) if os.path.splitext(img)[1] == '.avi' or os.path.splitext(img)[1] =='.mp4' or os.path.splitext(img)[1] =='.wmv']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), videofile))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(videofile))
        exit()
        
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    #改
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(save_file,x.split("/")[-1]))
    # print(det_names[0])
    
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    
    cam = cv2.VideoCapture(videofile)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    #读取视频框架的大小
    ret_val, input_image = cam.read()
    
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ending_frame = video_length
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourc=cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter(det_names[0],fourcc, input_fps, (input_image.shape[1], input_image.shape[0]))
    
    out = cv2.VideoWriter(det_names[0],fourc, input_fps, (int(cam.get(3)), int(cam.get(4))))
    # print(det_names[0])
    
    assert cam.isOpened(), 'Cannot capture source'
    # print(ending_frame)
    frames = 0
    start = time.time()    
    while (cam.isOpened()) and ret_val == True and frames < ending_frame:
        
        if frames % 1 ==0:
                   

            img, orig_im, dim = prep_image(input_image, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)


            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            out.write(orig_im)
            print(frames)
            
        frames += 1
        ret_val, input_image = cam.read()
        # print(frames)

    return det_names[0]


if __name__ == "__main__":
    cc=draw_save("1.avi")

    print(cc)

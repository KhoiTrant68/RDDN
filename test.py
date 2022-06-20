import argparse
import os
from sre_parse import State
from traceback import print_tb
import urllib.request
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
from utils import *
import os
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image
from net import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def main():
    start_time = time.time()
    psnrs = []
    ssims = []
    file_names = []
    cuda = 1
    cudnn.benchmark = True
    print("==========> Setting GPU")


    print("==========> Building model")
    
    model = final_Net()

    checkpoint = torch.load('./checkpoints/best_psnr.pth')
    model = nn.DataParallel(model, device_ids=[i for i in range(1)]).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    
    #===== Load input image =====
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )
    transform_gt = transforms.Compose([transforms.ToTensor()])
    model.eval()
    file_dir = "/home/DeHaze/vehicle/out_haze"
    for _, _, files in os.walk(file_dir):  
        for i in range(len(files)):
            #print(files) #
            imagePath = file_dir+'/'+files[i]

            frame = Image.open(imagePath)#2

            # frame = frame.resize((550, 400))
            imgIn = transform(frame).unsqueeze_(0)

            #===== Test procedures =====
            varIn = Variable(imgIn) 
            with torch.no_grad():
                output = model(varIn)
            output = torch.clamp(output, 0., 1.)
            computer_psnr = 1
            if computer_psnr:
                label_imagePath = '/home/DeHaze/vehicle/data_test/'+ files[i][:-8] + '.jpg'
                # print(imagePath)
                gt_img = Image.open(label_imagePath)
                # gt_img = gt_img.resize((550, 400))
                label = transform_gt(gt_img).unsqueeze_(0)
                label = label.cuda()
                psnrs.extend(to_psnr(output, label))
                ssims.extend(to_ssim_skimage(output, label))
            prediction = output.data.cpu().numpy().squeeze().transpose((1,2,0))
            prediction = (prediction*255.0).astype("uint8")
            im = Image.fromarray(prediction)
            save_path = "./results"
            if  not os.path.exists(save_path):
                os.makedirs(save_path)
            im.save(save_path+"/"+files[i])
            file_names.append(files[i])
    end_time = time.time() - start_time
    print(end_time)
    if computer_psnr:
        psnr_mean = sum(psnrs) / len(psnrs)
        print(psnr_mean)
        ssim_mean = sum(ssims) / len(ssims)
        print(ssim_mean)
        import pandas as pd 
        data = {"files":file_names,"psnr":psnrs, "ssim":ssims}
        test=pd.DataFrame(data,columns = ['files','psnr', 'ssim'])
        test.to_csv('results'+'.csv',index=False)

if __name__ == "__main__":
    os.system('clear')
    main()

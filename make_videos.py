#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:27:51 2022

@author: diego
"""

'''

import numpy as np
import cv2


def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (depth,width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(height):
        image_for_video=cv2.cvtColor(np.squeeze(volume[b,:,:]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()
    
    
    
def read_vid(path,s,e):
    cap = cv2.VideoCapture(path)
    reconstruction=[]
    while(cap.isOpened()):
        try:
          ret, frame = cap.read()
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          #cv2.imshow('frame',gray)
          reconstruction.append(gray[:,s:e])
          # if cv2.waitKey(1) & 0xFF == ord('q'):
          #   break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(reconstruction)

path='/home/diego/Downloads/RISLEY_COMPARISON/MODEL_EVALUATION_RISLEY_CAUCHY_TRANSMITTANCE_25_SIGMA_150_GT_MEDIAN_FILTER/comparative_reconstruction_3.avi'
path1='/home/diego/Downloads/MODEL_EVALUATION_REAL_GAUSSIAN_BLUE_NOISE_TRANSMITTANCE_25_SIGMA_200_GT_DENOISED/comparative_reconstruction_30.avi'
path3='/home/diego/Downloads/MODEL_EVALUATION_RANDOM_SUBSAMPLING_75/comparative_reconstruction_30.avi'
reconstruction_cauchy=read_vid(path,1050,2050)
reconstruction_best_theorical_case=read_vid(path1,1050,2050)
reconstruction_random=read_vid(path3,1050,2050)
GT=read_vid(path1,0,1000)

gap_gt=np.ones((100,512,1100)).astype(np.uint8)*255

GT_vid=np.concatenate((gap_gt,GT,gap_gt),axis=2)

gap=np.ones((100,512,50)).astype(np.uint8)*255
comparative_volume=np.concatenate((gap,reconstruction_best_theorical_case,
                                   gap,
                                   reconstruction_cauchy,
                                   gap,
                                   reconstruction_random,gap),axis=2)


sub_cauchy=read_vid(path,2100,3100)
sub_best_theorical_case=read_vid(path1,2100,3100)
sub_random=read_vid(path3,2100,3100)


sub_sampled_volume=np.concatenate((gap,sub_best_theorical_case,
                                   gap,
                                   sub_cauchy,
                                   gap,
                                   sub_random,gap),axis=2)


vert_gap=np.ones((100,70,3200)).astype(np.uint8)*255
vert_gap1=np.ones((100,10,3200)).astype(np.uint8)*255

total_vid=np.concatenate((vert_gap,GT_vid,
                          vert_gap,
                          comparative_volume,
                          vert_gap,
                          sub_sampled_volume,vert_gap1),axis=1)


#c=np.concatenate((comparative_volume[0:33,:,:],comparative_volume[48:90,:,:]))
c=total_vid[0:80,:,:]
make_video(c,'/home/diego/Downloads/SUPER_VIDEO/reconstructions')

'''


import cv2
import numpy as np


def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (depth,width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(height):
        image_for_video=cv2.cvtColor(np.squeeze(volume[b,:,:]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()
    
    
cap = cv2.VideoCapture('/home/diego/Downloads/SUPER_VIDEO/reconstructions.avi')
new_video=[]
while(True):
    try:
        # Capture frames in the video
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX
      
        # Use putText() method for
        # inserting text on video
        cv2.putText(gray, 
                    'Ground Truth', 
                    (1600-100, 40), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Reconstruction from Risley Beam Steering', 
                    (1600-300, 512+95), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Using Cauchy Distribution ', 
                    (1600-300+100, 512+95+30), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Reconstruction from Gaussian Blue Noise', 
                    (1600-1400, 512+95+20), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Reconstruction from Random Sampling', 
                    (1600+800-50, 512+95+20), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        
        cv2.putText(gray, 
                    'Sparsely Sampled Using Gaussian Blue Noise', 
                    (1600-1400, 590+512+95+20), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Sparsely Sampled Using Risley Beam Steering Pattern ', 
                    (1600-400, 590+512+95), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'with Cauchy Distribution', 
                    (1600-300+100, 590+512+95+30), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        cv2.putText(gray, 
                    'Sparsely Sampled Using Random Sampling', 
                    (1600+800-50, 590+512+95+20), 
                    font, 1, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_4)
        
        if(ret):
            new_video.append(gray)
        if(not ret):
            break
    except:
            break
  
final_video=np.array(new_video)    
  
# release the cap object
cap.release()
# close all windows

make_video(final_video,'/home/diego/Downloads/SUPER_VIDEO/reconstructions1')











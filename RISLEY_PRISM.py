#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:41:35 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def risley_optical_index_fused_silica(l):
    n=np.sqrt((((0.6961663)*l**2)/(l**2-0.0684043**2))+((0.4079426*l**2)/(l**2-0.1162414**2))+((0.8974794*l**2)/(l**2-9.896161**2))+1)
    return n

def generate_2D_pattern(tf,
                        PRF,
                        w,
                        w2,
                        w3,
                        w4,
                        a,
                        number_of_prisms,
                        n_prism,
                        expected_dims):
    #Number Of laser pulses in image capture time
    num_pulse=tf*PRF
    #laser spot number
    i=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
    #Time of laser Pulses
    t1=i*(1/PRF)
    #Angle of risley 1
    tr1= 2*np.pi*w*t1
    #Angle of risley 2
    tr2=2*np.pi*w2*t1
    #Angle of risley 3
    tr3=2*np.pi*w3*t1
    #Angle of risley 4
    tr4=2*np.pi*w4*t1
    
    
    
    
    
    n11=np.array([list(-np.cos(tr1)*np.sin(a)),
                  list(-np.sin(tr1)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr1.shape[0]))])
    
    n12=np.array([0*tr1,
                  0*tr1,
                  np.ones(tr1.shape[0])])
    n21=np.array([0*tr2,
                  0*tr2,
                  np.ones(tr2.shape[0])])
    n22=np.array([list(np.cos(tr2)*np.sin(a)),
                  list(np.sin(tr2)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr2.shape[0]))])
    
    n31=np.array([list(-np.cos(tr3)*np.sin(a)),
                  list(-np.sin(tr3)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr3.shape[0]))])
    n32=np.array([0*tr3,
                  0*tr3,
                  np.ones(tr3.shape[0])])
    
    n41=np.array([0*tr4,
                  0*tr4,
                  np.ones(tr4.shape[0])])
    n42=np.array([list(np.cos(tr4)*np.sin(a)),
                  list(np.sin(tr4)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr4.shape[0]))])
    
    N=[n11,n12,n21,n22,n31,n32,n41,n42]
    beam_a=[np.array([0,0,1])]
    
    
    dot_product=np.dot(beam_a[-1],N[1])
    
    
    for j in range(number_of_prisms*2):
        if((j+1)%2==0):
            dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
            
            escalar_part=(np.sqrt(1-(n_prism)**2*(1-(dot_product)**2))-((n_prism))*dot_product)
            vector_of_escalar=np.array([escalar_part,
                                        escalar_part,
                                        escalar_part])
            first_term_vec=(n_prism)*beam_a[-1]
            
            beam_a_new=first_term_vec+vector_of_escalar*N[j]
        else:
            if(j==0):
                dot_product=np.dot(beam_a[-1],N[j])
                escalar_part=(np.sqrt(1-((1/n_prism))**2*(1-(dot_product)**2))-((1/n_prism))*dot_product)
                vector_of_escalar=np.array([escalar_part,
                                            escalar_part,
                                            escalar_part])
                first_term_vec=np.array([list(((1/n_prism)*beam_a[-1]))]*tr1.shape[0])
                first_term_vec=np.transpose(first_term_vec)
                beam_a_new=first_term_vec+vector_of_escalar*N[j]
            else:
                dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
                
                escalar_part=(np.sqrt(1-(1/n_prism)**2*(1-(dot_product)**2))-((1/n_prism))*dot_product)
                vector_of_escalar=np.array([escalar_part,
                                            escalar_part,
                                            escalar_part])
                first_term_vec=(1/n_prism)*beam_a[-1]
                
                beam_a_new=first_term_vec+vector_of_escalar*N[j]
        beam_a.append(beam_a_new)
        
    # plt.rcParams["figure.figsize"] = (20,20)
    # plt.plot(beam_a[4][1],beam_a[4][0],'.')
    # plt.title('PATTERN USING 2 PRISMS')
    # plt.show() 
        
    # plt.rcParams["figure.figsize"] = (20,20)
    # plt.plot(beam_a[6][1],beam_a[6][0],'.')  
    # plt.title('PATTERN USING 3 PRISMS')
    # plt.show()   
        
 
            
      
    x_min=np.min(beam_a[-1][1])
    y_min=np.min(beam_a[-1][0])
    x_factor=np.abs((expected_dims[1]/2)/x_min)
    y_factor=np.abs((expected_dims[2]/2)/y_min)
    
    x=(beam_a[-1][1]*(x_factor))
    y=(beam_a[-1][0]*(y_factor+90))
    
    x = x+np.abs(np.min(x))
    y = y+np.abs(np.min(y))-27.5
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(x,y,'.')  
    plt.plot([0,0,1000,1000,0],[100,0,0,100,100],'r')
    plt.title('PATTERN USING 4 PRISMS')
    plt.show()  
    risley_pattern_2D=np.zeros((expected_dims[1],expected_dims[2]))
    for i in range(int(num_pulse)):
        discrete_points_x=round(x[i]) if (round(x[i])>0 and round(x[i])<expected_dims[1]) else expected_dims[1]-1 if round(x[i])>=expected_dims[1] else 0
        discrete_points_y=round(y[i]) if (round(y[i])>0 and round(y[i])<expected_dims[2]) else expected_dims[2]-1 if round(y[i])>=expected_dims[2] else 0
        
        risley_pattern_2D[discrete_points_x,discrete_points_y]=1
        
    transmittance=(risley_pattern_2D.sum()*100)/(expected_dims[1]*expected_dims[2])
    
    # cv2.imshow('Risley_pattern_2D',risley_pattern_2D.astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('TRANSMTTANCE C-SCAN',transmittance)
    return risley_pattern_2D,transmittance



def create_risley_pattern(expected_dims,
                          line_width,
                          start_wavelength,
                          tf,
                          PRF,
                          w,
                          w2,
                          w3,
                          w4,
                          a,
                          number_of_prisms):
    
    mask_risley=np.zeros(expected_dims)
    transmittance_list=[]
    for i in range(1,expected_dims[0]+1):
        print(i)
        
        #Risley optical index fused silica
        n_prism=risley_optical_index_fused_silica((i*line_width+start_wavelength)/1000)
        #print('Risley optical index fused silica ',n)
        mask_2D,transmittance=generate_2D_pattern(tf,
                            PRF+(i*(512/50)),
                            w,
                            w2,
                            w3,
                            w4,
                            a,
                            number_of_prisms,
                            n_prism,
                            expected_dims) 
        transmittance_list.append(transmittance)
        mask_risley[i-1,:,:]=mask_2D
        
    print('MEAN TRANSMTTANCE C-SCAN',np.mean(transmittance_list))
    transmittance_list=[]
    for m in range(100):
        transmittance_list.append((mask_risley[:,:,m].sum()*100)/(expected_dims[0]*expected_dims[1]))
    print('MEAN TRANSMTTANCE B-SCAN',np.mean(transmittance_list))
    total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
    print('TOTAL TRANSMITTANCE',total_transmittance)
    return mask_risley.astype(np.uint8)

def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()






number_of_prisms=4
#Laser Pulse Rate
PRF=19990000#1999000
#Image Capture Time 0.003
tf=0.016

#angular speed risley 1 rotations per sec
w=4000
#angula speed risley 2 rotations per sec
w2=(w/0.09)

#angula speed risley 2 rotations per sec
w3=(-w/0.09)

#angula speed risley 2 rotations per sec
w4=(-w/0.065)

a=1*(10*np.pi/180)    
expected_dims=(512,1000,100)   


band_width=176
line_width=band_width/expected_dims[0]
start_wavelength=962

mask_risley=create_risley_pattern(expected_dims,
                          line_width,
                          start_wavelength,
                          tf,
                          PRF,
                          w,
                          w2,
                          w3,
                          w4,
                          a,
                          number_of_prisms)


# from skimage import data
# import napari

# viewer = napari.view_image(magnitude_spectrum)

# img=mask
# f = np.fft.fft2(img.astype(np.float32))
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# for i in range(512):
#     plt.subplot(121),plt.imshow(img[i,:,:], cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(magnitude_spectrum[i,:,:], cmap = 'gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     plt.show()

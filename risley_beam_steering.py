#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:42:09 2022

@author: diego
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def risley_optical_index_fused_silica(l):
    n=np.sqrt((((0.6961663)*l**2)/(l**2-0.0684043**2))+((0.4079426*l**2)/(l**2-0.1162414**2))+((0.8974794*l**2)/(l**2-9.896161**2))+1)
    return n

def get_risley_scan_pattern(PRF,tf,w,phi,n,risley_angle):
    #Number Of laser pulses in image capture time
    num_pulse=tf*PRF
    #laser spot number
    i=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
    #Time of laser Pulses
    t1=i*(1/PRF)
    #Angle of risley 1
    angle_risley_1= w*t1*2*np.pi
    #Angle of risley 2
    angle_risley_2=2*np.pi*phi*t1+np.pi
    
    first_deflection_angle=math.asin((1/n)*math.sin(risley_angle))
    
    Bxi=first_deflection_angle*np.cos(angle_risley_1)
    Byi=first_deflection_angle*np.sin(angle_risley_1)
    
    gamma_xi=first_deflection_angle*np.cos(angle_risley_2)
    gamma_yi=first_deflection_angle*np.sin(angle_risley_2)
    
    x=Bxi+gamma_xi
    y=Byi+gamma_yi
    
    plt.plot(x,y,'-o')
    plt.show()
    
    
#Laser Pulse Rate
PRF=200000
#Image Capture Time
tf=0.0030

#angular speed risley 1 rotations per sec
w=-400
#angula speed risley 2 rotations per sec
phi=w/0.1

risley_angle=1*(np.pi/180)




for i in range(875,1235,10):
    print(i)
    #Risley optical index fused silica
    n=risley_optical_index_fused_silica(i)
    get_risley_scan_pattern(PRF,tf,w,phi,n,risley_angle)
    
    
    
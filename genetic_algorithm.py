#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:51:48 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from numpy import ndarray
from Generate_shifting_blue_noise import generate_shifting_blue_noise,generate_binary_blue_noise_mask,generate_shifting_blue_noise_all_directions
import cv2



expected_dimensions=(1000,100)

expected_dimensions=(512,1000)
blue=np.load('./blue_noise_cubes/bluenoise1024.npy')
mask=generate_binary_blue_noise_mask(blue,subsampling_percentage=0.75)
mask=mask[0:expected_dimensions[0],0:expected_dimensions[1]]
pattern=mask

# def create_random_mask(sub_sampling_percentage,expected_dims):
#     if(sub_sampling_percentage>0):
#         random_mask= np.random.choice([1, 0], size=expected_dims, p=[1-sub_sampling_percentage, sub_sampling_percentage])
#         total=random_mask.sum()
#         missing_data=(100-(total*100)/(random_mask.shape[0]*random_mask.shape[1]))
#         print(missing_data)
#     else:
#         random_mask=np.ones(expected_dims)
#     return random_mask

# pattern=create_random_mask(0.75,expected_dimensions)
# plt.imshow(pattern)
# plt.show()
def analyze_pattern(pattern):
    # cv2.imshow('BEST PATTERN',pattern)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    DFT=np.fft.fftshift(np.fft.fft2(pattern))/float(np.size(pattern));
    Height,Width=pattern.shape;
    ShiftY,ShiftX=(int(Height/2),int(Width/2));
    
    
    plt.imshow(np.abs(DFT),
                cmap="viridis",
                interpolation="nearest",
                vmin=0.0,
                vmax=np.percentile(np.abs(DFT),99))
                #extent=(-ShiftX-0.5,Width-ShiftX-0.5,-ShiftY+0.5,Height-ShiftY+0.5));
    plt.show()
def create_low_frecuencies_mask(expected_dimensions,axesLength,additional_mask=None,use_additional_mask=False):
    center_coordinates = (expected_dimensions[1]//2,expected_dimensions[0]//2)
      
    #axesLength = axesLength(20, 200)
      
    angle = 0
      
    startAngle = 0
      
    endAngle = 360
       
    # Red color in BGR
    color = (255, 255, 255)
       
    # Line thickness of 5 px
    thickness = -1
    
    
    
    mask_low_frecuency = np.zeros(expected_dimensions, dtype=np.uint8)
    mask_low_frecuency = cv2.ellipse(mask_low_frecuency,
                                     center_coordinates,
                                     axesLength,
                                     angle,
                                     startAngle,
                                     endAngle,
                                     color,
                                     thickness)//255
    mask_high_frecuency=np.logical_not(mask_low_frecuency)*1
    
    
    if(use_additional_mask):
        mask_low_frecuency_sub=np.logical_and(mask_low_frecuency,np.logical_not(additional_mask))*1
        lf_factor=np.sum(mask_low_frecuency)
        lf_s_factor=np.sum(mask_low_frecuency_sub)
        
        return mask_low_frecuency,mask_low_frecuency_sub,lf_factor,lf_s_factor
        
    hf_factor=np.sum(mask_high_frecuency)
    lf_factor=np.sum(mask_low_frecuency)

    return mask_low_frecuency,mask_high_frecuency,hf_factor,lf_factor


    
    
    
# f = np.fft.fft2(pattern.astype(np.float32))
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.imshow(magnitude_spectrum,cmap='hot')
# plt.show()




# start_x=5
# start_y=50
# step_x=1
# step_y=15
# low_frecuency_scores=[]
# for i in range(10):
#     if(i==0):
#         mask_low_frecuency,mask_high_frecuency,hf_factor,lf_factor=create_low_frecuencies_mask(expected_dimensions,
#                                                                                                (start_x+(i*step_x),start_y+(i*step_y)))
#         low_frecuency=np.multiply(mask_low_frecuency,magnitude_spectrum)
#         low_frecuency_scores.append((np.sum(low_frecuency)/lf_factor))
#         plt.imshow(mask_low_frecuency,cmap='gray')
#         plt.show()
#     else:
#         mask_low_frecuency,mask_low_frecuency_sub,lf_factor,lf_s_factor=create_low_frecuencies_mask(expected_dimensions,
#                                                                                                (start_x+(i*step_x),start_y+(i*step_y)),
#                                                                                                mask_low_frecuency,
#                                                                                                True)
#         low_frecuency=np.multiply(mask_low_frecuency_sub,magnitude_spectrum)
#         low_frecuency_scores.append((np.sum(low_frecuency)/lf_s_factor))
        
#         plt.imshow(mask_low_frecuency_sub,cmap='gray')
#         plt.show()
    
    
    
    
# high_frecuency=np.multiply(mask_high_frecuency ,magnitude_spectrum)
# low_frecuency=np.multiply(mask_low_frecuency,magnitude_spectrum)
# plt.imshow(np.abs(low_frecuency),
#             cmap="viridis",
#             interpolation="nearest",
#             vmin=0.0,
#             vmax=np.percentile(np.abs(low_frecuency),99))

# score=-np.log((np.sum(high_frecuency)/hf_factor)/(np.sum(low_frecuency)/lf_factor))

def compute_bluness(pattern):
    # plt.imshow(mask_low_frecuency)
    # plt.show()

    pattern=pattern-np.mean(pattern)
    # plt.imshow(pattern)
    # plt.show()
    f = np.fft.fft2(pattern.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    # magnitude_spectrum[np.isneginf(magnitude_spectrum)]=0
    plt.imshow(20*np.log(np.abs(fshift)))#[475:525,40:60]
    plt.show()
    
    
    
    # fshift[475:525,40:60] = 0
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img_back = np.real(img_back)
    #img_back_binarized=cv2.threshold(img_back, 128, 255, cv2.THRESH_BINARY)[1]
    
    # pattern=img_back
    # f = np.fft.fft2(pattern.astype(np.float32))
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))
    # magnitude_spectrum[np.isneginf(magnitude_spectrum)]=0
    # plt.imshow(magnitude_spectrum)#[475:525,40:60]
    # plt.show()
    
    N=10
    start_x=5
    start_y=50
    step_x=1
    step_y=15
    low_frecuency_scores=[]
    for i in range(N):
        if(i==0):
            mask_low_frecuency,mask_high_frecuency,hf_factor,lf_factor=create_low_frecuencies_mask(expected_dimensions,
                                                                                                   (start_x+(i*step_x),start_y+(i*step_y)))
            #mask_low_frecuency[expected_dimensions[0]//2,expected_dimensions[1]//2]=0    
            low_frecuency=np.multiply(mask_low_frecuency,magnitude_spectrum)
            low_frecuency_scores.append((np.sum(low_frecuency)/lf_factor))
            #mask_low_frecuency[expected_dimensions[0]//2,expected_dimensions[1]//2]=1
            # plt.rcParams["figure.figsize"] = (50,100)
            # plt.imshow(mask_low_frecuency,cmap='gray')
            # plt.show()
        else:
            mask_low_frecuency,mask_low_frecuency_sub,lf_factor,lf_s_factor=create_low_frecuencies_mask(expected_dimensions,
                                                                                                   (start_x+(i*step_x),start_y+(i*step_y)),
                                                                                                   mask_low_frecuency,
                                                                                                   True)
            low_frecuency=np.multiply(mask_low_frecuency_sub,magnitude_spectrum)
            low_frecuency_scores.append((np.sum(low_frecuency)/lf_s_factor))
            
            # plt.rcParams["figure.figsize"] = (50,100)
            # plt.imshow(mask_low_frecuency_sub,cmap='gray')
            # plt.show()
    
    
    
    mask_low_frecuency,mask_high_frecuency,hf_factor,lf_factor=create_low_frecuencies_mask((1000,100),(20, 200))

    
    high_frecuency=np.multiply(mask_high_frecuency ,magnitude_spectrum)
    
    # plt.imshow(mask_high_frecuency,cmap='gray')
    # plt.show()
    
    low_frecuency=np.multiply(mask_low_frecuency,magnitude_spectrum)
    
    # plt.imshow(mask_low_frecuency,cmap='gray')
    # plt.show()
    
    score_list=[(low_frecuency_scores[ind+1]/low_frecuency_scores[ind]) for ind in range(len(low_frecuency_scores)-1)]    
    
    hf=(np.sum(high_frecuency)/hf_factor)
    lf=(np.sum(low_frecuency)/lf_factor)
    part_1=(np.sum(score_list))/(N-1)
    part_2=(hf/lf)
    score=(part_1+part_2)#the higher the better. For minimize, add - symbol
    return score

    # DFT=np.fft.fftshift(np.fft.fft2(pattern))/float(np.size(pattern));
    # Height,Width=pattern.shape;
    # ShiftY,ShiftX=(int(Height/2),int(Width/2));
    # plt.rcParams["figure.figsize"] = (8,8)
    
    # plt.imshow(np.abs(DFT),
    #             cmap="viridis",
    #             interpolation="nearest",
    #             vmin=0.0,
    #             vmax=np.percentile(np.abs(DFT),99))
    #             #extent=(-ShiftX-0.5,Width-ShiftX-0.5,-ShiftY+0.5,Height-ShiftY+0.5));
    # plt.show()
    # print('yupi')
def plot_fn(x,y,title,fontsize,xlabel,ylabel,img_size=(20,20),draw_FOV=False):
    plt.rcParams["figure.figsize"] = img_size
    plt.plot(x,y,'.')
    if(draw_FOV):
        plt.plot([0,0,1000,1000,0],[100,0,0,100,100],'r')
    plt.title(title,fontsize=fontsize)
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
def generate_2D_pattern(w1,
                        w2,
                        w3,
                        w4,
                        tf=0.16,
                        PRF=199900,
                        a=10*(np.pi/180),
                        number_of_prisms=4,
                        n_prism=1.444,
                        expected_dims=(512,1000,100),
                        pattern_for_distance=False,
                        plot_mask=False):
    

    #Number Of laser pulses in image capture time
    num_pulse=tf*PRF
    #laser spot number
    i=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
    #Time of laser Pulses
    t1=i*(1/PRF)
    #Angle of risley 1
    tr1= 2*np.pi*w1*t1
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
    n21=n12
    n22=np.array([list(np.cos(tr2)*np.sin(a)),
                  list(np.sin(tr2)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr2.shape[0]))])
    
    n31=np.array([list(-np.cos(tr3)*np.sin(a)),
                  list(-np.sin(tr3)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr3.shape[0]))])
    n32=n12
    
    n41=n12
    n42=np.array([list(np.cos(tr4)*np.sin(a)),
                  list(np.sin(tr4)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr4.shape[0]))])
    
    N=[n11,n12,n21,n22,n31,n32,n41,n42]
    beam_a=[np.array([0,0,1])]
    
    
    dot_product=np.dot(beam_a[-1],N[1])
    


    for j in range(number_of_prisms*2):
        if((j+1)%2==0):
            #dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
            dot_product=np.einsum('ij,ji->i', np.transpose(beam_a[-1]),N[j])
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
                #dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
                dot_product=np.einsum('ij,ji->i', np.transpose(beam_a[-1]),N[j])
                escalar_part=(np.sqrt(1-(1/n_prism)**2*(1-(dot_product)**2))-((1/n_prism))*dot_product)
                vector_of_escalar=np.array([escalar_part,
                                            escalar_part,
                                            escalar_part])
                first_term_vec=(1/n_prism)*beam_a[-1]
                
                beam_a_new=first_term_vec+vector_of_escalar*N[j]
        beam_a.append(beam_a_new)
    
    central_d=12
    
    A1=np.abs(central_d/beam_a[1][2])*beam_a[1]
    A2=np.abs(76/beam_a[2][2])*beam_a[2]
    A2=A1+A2
    
    
    r=np.sqrt(A2[0,:]**2+A2[1,:]**2)
    T_p=r*np.tan(a)*np.cos(np.abs(tr1-tr2))
    
    A3=np.abs((central_d+T_p)/beam_a[3][2])*beam_a[3]
    A3=A3+A2
    
    d_3=188-A3[2,:]
    
    A4=np.abs(d_3/beam_a[4][2])*beam_a[4]
    
    A4=A4+A3
    
    r1=np.sqrt(A4[0,:]**2+A4[1,:]**2)
    T_p1=r1*np.tan(a)*np.cos(np.abs(tr2-tr3))
    A5=np.abs((central_d+T_p1)/beam_a[5][2])*beam_a[5]
    
    A5=A5+A4
    
    d_5=288-A5[2,:]
    A6=np.abs(d_5/beam_a[6][2])*beam_a[6]
    A6=A6+A5
    
    
    r2=np.sqrt(A6[0,:]**2+A6[1,:]**2)
    T_p2=r2*np.tan(a)*np.cos(np.abs(tr3-tr4))
    A7=np.abs((central_d+T_p2)/beam_a[7][2])*beam_a[7]
    
    A7=A7+A6
    
    d_7=350-A7[2,:]
    A8=np.abs(d_7/beam_a[8][2])*beam_a[8]
    A8=A8+A7

    x_max=np.max(A8[1,:])
    y_max=np.max(A8[0,:])
    x_factor=np.abs((expected_dims[1]/2)/x_max)
    y_factor=np.abs((expected_dims[2]/2)/y_max)
    
    x=(A8[1,:]*(x_factor+5.5))
    y=(A8[0,:]*(y_factor+0.35))
    
    x = x+500
    y = y+50

    
    risley_pattern_2D=np.zeros((expected_dims[1],expected_dims[2]))
    
    aa=ndarray.round(x)
    bb=ndarray.round(y)
    

    keep_x_coordinate=np.logical_and(aa >= 0 , aa < expected_dims[1])
    keep_y_coordinate=np.logical_and(bb >= 0 , bb < expected_dims[2])
    
    remove_coordinates=np.logical_not(np.logical_and(keep_x_coordinate,keep_y_coordinate))
    bb=[d for (d, remove) in zip(bb, remove_coordinates) if not remove]
    aa=[d for (d, remove) in zip(aa, remove_coordinates) if not remove]


    coordinates=np.array((np.array(aa).astype(int),np.array(bb).astype(int)))
    
    
    
    risley_pattern_2D[tuple(coordinates)] = 255
    
    if(plot_mask):
        plot_fn(x=A8[1,:],y=A8[0,:],title='PATTERN USING 4 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        plot_fn(x,
                y,
                title=f'FINAL PATTERN USING 4 PRISM \n w1={w1} rps,w2={w2} rps,w3={w3} rps,w4={w4} rps',
                fontsize=80,
                xlabel='Pixels',
                ylabel='Pixels',
                img_size=(80,25),
                draw_FOV=True)
    if(pattern_for_distance):
        return A8
    cv2.imwrite('BEST_PATTERN.jpeg', risley_pattern_2D*255)
    return risley_pattern_2D
# pattern=generate_2D_pattern(w1=9990,w2=9990/0.09,w3=9990/-0.09,w4=9990/0.065)
# pattern=generate_2D_pattern(9990,
#                             111000,
#                             12333,
#                             119538,
#                             plot_mask=True)

pattern=generate_2D_pattern(w1=-3428.22250471,
                            w2=-1101.58077614,
                            w3=-1530.3051244,
                            w4=4721.40253875,plot_mask=True)

# cv2.imwrite('pattern.jpeg',pattern)
compute_bluness(pattern)


# pattern=generate_2D_pattern(w1=6255.54063372,w2=-2020.10559296,w3=-1227.16073769,w4=1227.40445477,plot_mask=True)
# compute_bluness(pattern)

########################################################################################################################################
'''
n_pop=1000
n_bits=32
n_iter=100
# crossover rate
r_cross = 0.8


# define range for input
bounds = [[-200000, 200000], [-200000, 200000],[-200000, 200000],[-200000, 200000]]

# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded


def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if np.random.rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = np.random.randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if np.random.rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]


# objective function
def objective(x):
    pattern=generate_2D_pattern(w1=x[0],w2=x[1],w3=x[2],w4=x[3])
    return compute_bluness(pattern)
# 	return x[0]**2.0 + x[1]**2.0

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best_eval = 1000000000
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(c) for c in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                angular_speeds= decode(bounds, n_bits, pop[i])
                best_pattern=generate_2D_pattern(angular_speeds[0],
                                                 angular_speeds[1],
                                                 angular_speeds[2],
                                                 angular_speeds[3],plot_mask=True)
                analyze_pattern(best_pattern)
                print(">%d, new best f(%s) = %.10f" % (gen, angular_speeds, scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
        	# get selected parents in pairs
        	p1, p2 = selected[i], selected[i+1]
        	# crossover and mutation
        	for c in crossover(p1, p2, r_cross):
        		# mutation
        		mutation(c, r_mut)
        		# store for next generation
        		children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

# perform the genetic algorithm search
best, score = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (decode(bounds, n_bits, best), score))
'''
########################################################################################################################################
from scipy.linalg import get_blas_funcs
def fitness_func(solution, solution_idx):
    pattern=generate_2D_pattern(w1=solution[0],
                                w2=solution[1],
                                w3=solution[2],
                                w4=solution[3],
                                pattern_for_distance=False,
                                plot_mask=False)
    score=compute_bluness(pattern)
    
    # pattern=np.transpose(pattern[0:2,:])
    
    
    # gemm=get_blas_funcs("gemm", [pattern, pattern.T])
    # term1=gemm(1, pattern, pattern.T).astype(np.float16)
    # term2=np.sum(pattern**2, axis=1, keepdims=True).astype(np.float16)
    # term3=np.sum(pattern**2, axis=1).astype(np.float16)
    # num_test = pattern.shape[0]
    # #dists = np.zeros((num_test, num_test)) 
    # dists = np.sqrt((-2 * term1) + term2 + term3)
    
    # dists=euclidean_distances(pattern)
    # suma=np.sum(dists,axis=1)
    
    return score
#score=fitness_func([-14463.702398,    -9660.09853304, 196718.5323155 , 141915.63981462], solution_idx=None)
import pygad
bounds = [[-5000, 5000], [-5000, 5000],[-5000, 5000],[-5000, 5000]]
fitness_function = fitness_func

num_generations = 100
num_parents_mating = 4

sol_per_pop = 100
num_genes = 4

init_range_low = -5000
init_range_high = 5000

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

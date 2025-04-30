import cv2
import numpy as np

import tqdm

import sys
import os

POINTS = 100000

MIN_COEFF = 400
MAX_COEFF = 5000

FRAMES = 500

MAX_DIST = 2

UPSCALING = 1.5
SMOOTHING_WINDOW = 5

def read_tour(tour_f):
    with open(tour_f) as f:
        f_it = iter(f)
        next(f_it)

        tour = []
        for l in f_it:
            if not l:
                break

            s,e,w = map(int,l.split())
            tour.append([s,e,w])
        
        return np.array(tour)

if __name__ == "__main__":
    fourier_file, tour_file, res, output_root = sys.argv[1:5]
    res_x, res_y = map(int,res.split("x"))
    res_x, res_y = int(res_x*UPSCALING), int(res_y*UPSCALING)
    center_x, center_y = res_x // 2, res_y // 2

    path = os.path.dirname(output_root)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(fourier_file) as ff:
        coeffs = [(int(ci),float(cx),float(cy)) for ci,cx,cy in map(str.split,ff)]

    tour = read_tour(tour_file)
    facepoints = np.interp(np.linspace(0,len(tour),POINTS,endpoint=False),np.arange(len(tour)),tour[:,2] <= MAX_DIST)
    facepoints = np.convolve(facepoints,np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW,"same")
    facepoints = np.linspace(0,1,POINTS,endpoint=False)[facepoints >= 0.999]

    
    coeff_i, *coeffs =  zip(*coeffs)
    coeff_i = np.array(coeff_i)
    coeffs = np.array(coeffs)

    use_coeffs = (coeff_i <= MAX_COEFF) & (-MAX_COEFF <= coeff_i) & (coeff_i != 0)
    coeff_i = coeff_i[use_coeffs]
    coeffs = coeffs[:,use_coeffs]

    coscoeff = np.cos(2*np.pi*facepoints[None,:]*coeff_i[:,None])
    sincoeff = np.sin(2*np.pi*facepoints[None,:]*coeff_i[:,None])

    X, Y = center_x, center_y
    last_coeff = 0
    for COEFF in tqdm.tqdm(range(MIN_COEFF,MAX_COEFF,(MAX_COEFF-MIN_COEFF)//FRAMES)):
        img =  np.zeros((res_x, res_y), dtype=np.uint8)

        range_1 = slice(MAX_COEFF-COEFF,MAX_COEFF-last_coeff)
        range_2 = slice(MAX_COEFF+last_coeff,MAX_COEFF+COEFF)

        last_coeff = COEFF

        X += (np.sum(coscoeff[range_1]*coeffs[0][range_1,None] - sincoeff[range_1]*coeffs[1][range_1,None],axis=0) + 
                np.sum(coscoeff[range_2]*coeffs[0][range_2,None] - sincoeff[range_2]*coeffs[1][range_2,None],axis=0))*UPSCALING
        Y += (np.sum(sincoeff[range_1]*coeffs[0][range_1,None] + coscoeff[range_1]*coeffs[1][range_1,None],axis=0) +
                np.sum(sincoeff[range_2]*coeffs[0][range_2,None] + coscoeff[range_2]*coeffs[1][range_2,None],axis=0))*UPSCALING

        X_pix = np.clip(np.round(X),0,res_x-1).astype(dtype=int)
        Y_pix = np.clip(np.round(Y),0,res_y-1).astype(dtype=int)

        img[X_pix,Y_pix] = 255

        cv2.imwrite("{}{:06}.png".format(output_root,COEFF),img)
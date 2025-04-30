import cv2
import numpy as np

import tqdm

import multiprocessing as mp

import sys
import os

POINTS = 1000000
FRAMES = 5000
MAX_COEFF = 400

MAX_DIST = 2
UPSCALING = 1.5
ZOOM = 1

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

def draw_line(p1,p2,buffer,color=[255,255,255]):
    tc_dist = int(np.sum(np.abs(p2-p1)))
    points = np.round(np.linspace(p1,p2,tc_dist))
    p_X = np.clip(points[:,0],0,buffer.shape[0]-1).astype(dtype=int)
    p_Y = np.clip(points[:,1],0,buffer.shape[1]-1).astype(dtype=int)

    buffer[p_X,p_Y,:] = color

if __name__ == "__main__":
    fourier_file, tour_file, res, output_root = sys.argv[1:5]

    path = os.path.dirname(output_root)
    if not os.path.exists(path):
        os.makedirs(path)

    res_x, res_y = map(int,res.split("x"))
    res_x, res_y = int(res_x*UPSCALING), int(res_y*UPSCALING)
    center_x, center_y = res_x // 2, res_y // 2

    zoom_res_x, zoom_res_y = res_x//ZOOM, res_y//ZOOM

    with open(fourier_file) as ff:
        coeffs = [(int(ci),float(cx),float(cy)) for ci,cx,cy in map(str.split,ff)]

    tour = read_tour(tour_file)
    facepoints = np.interp(np.linspace(0,len(tour),POINTS,endpoint=False),np.arange(len(tour)),tour[:,2] <= MAX_DIST)
    facepoints = np.linspace(0,1,POINTS,endpoint=False)[facepoints >= 0.999]

    
    coeff_i, *coeffs =  zip(*coeffs)
    coeff_i = np.array(coeff_i)
    coeffs = np.array(coeffs)

    use_coeffs = (coeff_i <= MAX_COEFF) & (-MAX_COEFF <= coeff_i) & (coeff_i != 0)
    coeff_i = coeff_i[use_coeffs]
    coeffs = coeffs[:,use_coeffs]

    coscoeff = np.cos(2*np.pi*facepoints[None,:]*coeff_i[:,None])
    sincoeff = np.sin(2*np.pi*facepoints[None,:]*coeff_i[:,None])

    X = coscoeff*coeffs[0][:,None] - sincoeff*coeffs[1][:,None]
    Y = sincoeff*coeffs[0][:,None] + coscoeff*coeffs[1][:,None]

    XY = np.stack((X,Y),axis=-1)*UPSCALING
    XY_sum = np.round(np.sum(XY,axis=0))

    X_pix = np.clip(XY_sum[:,0]+ center_x,0,res_x-1).astype(dtype=int) 
    Y_pix = np.clip(XY_sum[:,1]+ center_y,0,res_y-1).astype(dtype=int) 

    print(XY.shape)


    def _f(i):
        img = np.zeros((res_x,res_y,3), dtype=np.uint8)
        img[X_pix[:i+1],Y_pix[:i+1],:] = [255,255,255]
        
        point = np.array([center_x,center_y])
        for j in range(XY.shape[0]):
            coeff_idx = (MAX_COEFF-1) + ((j%2)+(j%2-1))*((j+1)//2)
            n_point = point + XY[coeff_idx,i]
            draw_line(point, n_point, img, color=[0,max(int(255-j),0),0])
            point = n_point

        point = point.astype(dtype=int)
        if point[0]-zoom_res_x//2 < 0:
            frame_x1 = 0
        elif point[0]+zoom_res_x//2 > res_x:
            frame_x1 = res_x-zoom_res_x
        else:
            frame_x1 = point[0]-zoom_res_x//2
        frame_x2 = frame_x1+zoom_res_x

        if point[1]-zoom_res_y//2 < 0:
            frame_y1 = 0
        elif point[1]+zoom_res_y//2 > res_y:
            frame_y1 = res_y-zoom_res_y
        else:
            frame_y1 = point[1]-zoom_res_y//2
        frame_y2 = frame_y1+zoom_res_y

        cv2.imwrite("{}{:06}.png".format(output_root,i//(XY.shape[1]//FRAMES)),img[frame_x1:frame_x2,frame_y1:frame_y2])

    with  mp.Pool() as p:
        t = p.imap_unordered(_f,range(0,XY.shape[1],XY.shape[1]//FRAMES))
        list(tqdm.tqdm(t,total=FRAMES))
        

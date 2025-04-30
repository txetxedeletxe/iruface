from iruface.tsputils import tsplib_points_from_coord

import cv2
import numpy as np

from argparse import ArgumentParser


def _build_parser():
    parser = ArgumentParser()
    parser.add_argument("image",help="Image to convert to TSP format.")

    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # Load and Process image
    img = cv2.imread(args.image,)
    img_bw = np.max(img,axis=2) # Make monochrome
    
    x,y = np.nonzero(img_bw)
    xy = np.stack((x,y),axis=1)

    # Get tsp format points
    tsp = tsplib_points_from_coord(xy)
    print(tsp)


    


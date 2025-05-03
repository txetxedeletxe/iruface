from iruface.tsputils import read_tsplib_points, read_tsplib_tour

import cv2
import numpy as np

import colorsys

from argparse import ArgumentParser

import os
    
def _build_parser():
    parser = ArgumentParser()

    # TODO add save image feature
    parser.add_argument("tsp",help=".tsp file containing the points the curve")
    parser.add_argument("tour",help=".sol file containing a tour across the points of the curve")
    parser.add_argument("-o","--output",dest="output",help="Output file path")

    image_parser = parser.add_argument_group("IMAGE")
    image_parser.add_argument("--continuity_threshold",type=float,default=None,
    help="""Max distance between adjacent points in the tour, value must be greater than one.
    Edges between points at greater distances are not drawn. By default all edges are drawn""")
    image_parser.add_argument("--rainbow_trace",action="store_true",
    help="""Shift color during the tracing of the tour, increasing hue from start to finish""")

    image_parser.set_defaults(rainbow_trace=False)

    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    points = read_tsplib_points(args.tsp)
    tour = read_tsplib_tour(args.tour)

    img =  np.zeros([*(np.max(points,axis=0)+1),3], dtype=np.uint8)

    for i, (p1, p2, w) in enumerate(tour):
        if args.continuity_threshold is not None and w > args.continuity_threshold:
            continue

        (x1,y1), (x2,y2) = points[p1], points[p2]
        diff = abs(x2-x1) + abs(y2-y1)

        # Fill intermediate values
        for t in range(diff):
            x = x1*(1-t/diff) + x2*t/diff
            y = y1*(1-t/diff) + y2*t/diff

            if args.rainbow_trace:
                img[round(x),round(y),:] = np.round(np.array([*colorsys.hsv_to_rgb(0.9*i/len(tour),1,1)])*255).astype(np.uint8)
            else:
                img[round(x),round(y),:] = 255
    
    # Show or save image
    if args.output is None:
        cv2.imshow(os.path.basename(args.tsp).split(".")[0],img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output,img)
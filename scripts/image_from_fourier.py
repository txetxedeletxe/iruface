from iruface.fourier import evaluate_fourier_sum
from iruface.tsputils import read_tsplib_tour

import cv2
import numpy as np

from argparse import ArgumentParser
import os

def _build_parser():
    parser = ArgumentParser()

    parser.add_argument("fourier",help="File with fourier coefficients")
    parser.add_argument("-o","--output",dest="output",help="Output file path")

    quality_parser = parser.add_argument_group("QUALITY")
    quality_parser.add_argument("res",help="Resolution of the output image, must be expecified in 'WxH' format")
    quality_parser.add_argument("--img_scale",type=float,default=1,help="""Scale factor for the produced image""")
    quality_parser.add_argument("--points",type=int,default=int(1e4),
    help="""Amount of individual points along the curve to render. Higher values increase quality
    at the expense of requiring more resources. Should be set depending on 'res' argument and the complexity of the curve""")
    quality_parser.add_argument("--interpolate_points",type=int,default=int(1e5),
    help="""Total amount of points to interpolate between curve points. Higher values increase quality
    requireing less resources than 'points' parameter. This value should in most cases be larger than the 'points' parameter""")
    quality_parser.add_argument("--max_interpolate_deviation",type=float,default=2.0,
    help="""Maximum allowed statistical deviation between adjacent interpolated points.""")
    quality_parser.add_argument("--fourier_order",type=int,default=None,
    help="""Order of the fourier sum to use to approximate curve. By default all coefficients in the 'fourier' file are used""")
    
    smoothing_parser = parser.add_argument_group("SMOOTHING")
    smoothing_parser.add_argument("--tour",
    help="""File with TSP tour that the fourier coefficients correspond to.
    It is used to erase/smooth discontinuities in the traced curve. If not provided
    the rest of options on this group are ignored""")
    smoothing_parser.add_argument("--smoothing",type=int,default=5,
    help="""Size of the uniform smoothing filter. Higher values increase smoothness.
    A value of 1 applies no smoothing""")
    smoothing_parser.add_argument("--continuity_threshold",type=float,default=2,
    help="""Max distance between adjacent points in the tour, value must be greater than one.
    The greater the value, the larger the allowed discontinuities (before smoothing)""")
    smoothing_parser.add_argument("--smoothing_threshold",type=float,default=0.99,
    help="""Minimum strength of an interpolated point for it to be drawn in the image. 
    The threshold must be in [0,1] (and is usually close to 1)""")

    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    with open(args.fourier) as ff:
        coeffs = [(int(ci),float(cx),float(cy)) for ci,cx,cy in map(str.split,ff)]

    res_y, res_x = map(int,args.res.split("x"))
    center_x, center_y = res_x // 2, res_y // 2
    img = np.zeros((res_x, res_y), dtype=np.uint8) # Create image

    # FILTER TOUR
    curve_parameter = np.linspace(0,1,args.points,endpoint=False)
    if args.tour is not None:
        tour = read_tsplib_tour(args.tour)
        
        parameter_softmask = np.interp(np.linspace(0,len(tour),args.points,endpoint=False),np.arange(len(tour)),tour[:,2] <= args.continuity_threshold)
        parameter_softmask = np.convolve(parameter_softmask,np.ones(args.smoothing)/args.smoothing,"same")

        curve_parameter = curve_parameter[parameter_softmask >= args.smoothing_threshold]

    # COEFFICIENTS
    coeff_i, *coeffs =  zip(*coeffs)
    coeff_i = np.array(coeff_i) # Coefficient indices
    coeffs = np.array(coeffs) # Coefficients

    # Reduce order
    if args.fourier_order is not None:
        use_coeffs = (coeff_i <= args.fourier_order) & (-args.fourier_order <= coeff_i) & (coeff_i != 0)
        coeff_i = coeff_i[use_coeffs]
        coeffs = coeffs[:,use_coeffs]

    # Get fourier sum
    XY = evaluate_fourier_sum(curve_parameter,coeffs,coeff_i)
    XY = XY*args.img_scale + np.array([[center_x,center_y]])

    # Interpolate curve points
    # Obtain mask
    curve_diff = np.diff(curve_parameter)
    th_step = np.mean(curve_diff) + np.std(curve_diff)*args.max_interpolate_deviation # Threshold step

    curve_mask = np.concat(([1],curve_diff < th_step))
    
    # Obtain parameter values at which to interpolate
    interpolated_mask = np.interp(np.linspace(0,1,args.interpolate_points,endpoint=False),curve_parameter,curve_mask)
    interpolated_mask = interpolated_mask == 1
    interpolated_parameter = np.interp(np.linspace(0,1,args.interpolate_points,endpoint=False),curve_parameter,curve_parameter)[interpolated_mask]

    # interpolate
    X = np.interp(interpolated_parameter,curve_parameter,XY[:,0])
    Y = np.interp(interpolated_parameter,curve_parameter,XY[:,1])

    # Convert to pixel coordinates
    X_pix = np.clip(np.round(X),0,res_x-1).astype(dtype=int)
    Y_pix = np.clip(np.round(Y),0,res_y-1).astype(dtype=int)

    img[X_pix,Y_pix] = 255 # Set image pixels to white

    # Show or save image
    if args.output is None:
        cv2.imshow(os.path.basename(args.fourier).split(".")[0],img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output,img)

from iruface.fourier import fourier_coeff_from_curve
from iruface.tsputils import read_tsplib_tour, read_tsplib_points

import numpy as np

from argparse import ArgumentParser

def _build_parser(): # TODO implement mechanisms to personalize tour traversal
    parser = ArgumentParser()

    parser.add_argument("tsp",help=".tsp file containing the points the curve.")
    parser.add_argument("tour",help=".sol file containing a tour across the points of the curve.")

    coefficient_parser = parser.add_argument_group("COEFFICIENTS")
    coefficient_parser.add_argument("coeff_count",type=int,help="Amount of fourier coefficients to compute (coefficients s.t. -coeff_count <= i <= coeff_count, are computed).")
    coefficient_parser.add_argument("--origin",default=None,
    help="""Origin of the system to use when computing coefficients (essentially a translation of the 0-coefficient).
    Must be specified in "XxY" format, if not specified a suitable origin is estimated using points in the tsp file.""")
    coefficient_parser.add_argument("--cuadrature_nodes",type=int,default=int(1e4),
    help="""Amount of integration nodes to use when doing numerical cuadrature. Higher values produce better approximations of the coefficients.""")

    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # Open Files
    points = read_tsplib_points(args.tsp)
    tour = read_tsplib_tour(args.tour)

    # Find origin
    if args.origin:
        origin = np.array([*map(float,args.origin.split("x"))])
    else:
        origin = np.mean(points,axis=0)

    # Recenter points
    points = points - origin[None,:]

    # Get parametrized curve
    f_x, f_y = points[tour[:,0]].T
    f_x = np.interp(np.linspace(0,len(f_x),args.cuadrature_nodes,endpoint=False),np.arange(len(f_x)),f_x)
    f_y = np.interp(np.linspace(0,len(f_y),args.cuadrature_nodes,endpoint=False),np.arange(len(f_y)),f_y)

    curve = np.stack((f_x,f_y),axis=1)

    # Obtain fourier coefficients
    ci, (rc,ic) = fourier_coeff_from_curve(curve,args.coeff_count)

    # Print in csv format
    for i,re,img in zip(ci,rc,ic):
        print(i,re,img)



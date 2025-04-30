import io

import numpy as np

# Reading utilities
def read_tsplib_tour(tour_f):
    with open(tour_f) as f:
        f_it = iter(f)
        next(f_it) # Skip first line

        tour = []
        for l in f_it:
            if not l:
                break

            s,e,w = map(int,l.split())
            tour.append([s,e,w])
        
        return np.array(tour)

def read_tsplib_points(tsp_f):
    with open(tsp_f) as f:
        f_it = iter(f)
        while "NODE_COORD_SECTION" not in next(f_it): # Skip until coordinates
            pass

        points = []
        for l in f_it:
            if "EOF" in l:
                break

            _,x,y = map(int,l.split())
            points.append([x,y])

        return np.array(points)

# Creation utilities
def tsplib_points_from_coord(xy):
    sio = io.StringIO()
    
    # SPECIFICATION
    print("TYPE : TSP",file=sio)
    print(f"DIMENSION: {xy.shape[0]}", file=sio)
    print(f"EDGE_WEIGHT_TYPE : EUC_2D", file=sio)

    # NODE COORD
    print("NODE_COORD_SECTION",file=sio)
    for l,(x,y) in enumerate(xy):
        print(l,x,y,file=sio)

    print("EOF",file=sio) #EOF

    sio.seek(0)
    return sio.read()
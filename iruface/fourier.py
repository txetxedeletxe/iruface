import numpy as np

# TODO generalize to higher dimensions
# TODO improve efficiency with batching

def fourier_coeff_from_curve(curve_points : np.ndarray, coeff_count : int): 

    f_x, f_y = curve_points.T

    # Prepare trig function coeffs
    coscoeff = np.cos(-2*np.pi*np.linspace(0,1,len(curve_points),endpoint=False)[None,:]*np.arange(-coeff_count,coeff_count+1)[:,None])
    sincoeff = np.sin(-2*np.pi*np.linspace(0,1,len(curve_points),endpoint=False)[None,:]*np.arange(-coeff_count,coeff_count+1)[:,None])

    # This is analogous to cuadrature between 0 and 1 # TODO use trapezoidal rule
    rc = np.mean(coscoeff*f_x[None] - sincoeff*f_y[None],axis=1) # Real Coefficients
    ic = np.mean(coscoeff*f_y[None] + sincoeff*f_x[None],axis=1) # Imaginary Coefficients

    # Coefficient indices
    ci = np.arange(-coeff_count,coeff_count+1,dtype=int) 

    # $f_x(t) \simeq \sum_{i \in ci} cc(i)*cos(2*i*\pi*t)-sc(i)*sin(2*i*\pi*t) : t \in [0,1]$
    # $f_y(t) \simeq \sum_{i \in ci} sc(i)*cos(2*i*\pi*t)+cc(i)*sin(2*i*\pi*t) : t \in [0,1]$
    return ci, (rc, ic) 

def evaluate_fourier_sum(parameter : np.ndarray, coeffs : np.ndarray, coeff_idx : np.ndarray = None):

    if coeff_idx is None:
        coeff_count = (len(coeffs)-1)//2
        coeff_idx = np.arange(-coeff_count,coeff_count+1,dtype=int) 

    # Prepare trig function coeffs
    coscoeff = np.cos(2*np.pi*parameter[None,:]*coeff_idx[:,None])
    sincoeff = np.sin(2*np.pi*parameter[None,:]*coeff_idx[:,None])

    # Compute sum
    X = np.sum(coscoeff*coeffs[0][:,None] - sincoeff*coeffs[1][:,None],axis=0)
    Y = np.sum(sincoeff*coeffs[0][:,None] + coscoeff*coeffs[1][:,None],axis=0)

    return np.stack((X,Y),axis=1)


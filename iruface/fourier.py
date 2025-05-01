import numpy as np

try:
    import multiprocess as mp
except ImportError:
    pass

try:
    import tqdm
except ImportError:
    pass

import sys


# TODO generalize to higher dimensions

class FourierRuntime:
    batch_size = 4
    use_tqdm = True
    threads = None # i.e. use all threads

    def __init__(self,*,
            batch_size : int = None,
            use_tqdm : bool = None,
            threads : int = None):

        if batch_size is not None: self.batch_size = batch_size
        if use_tqdm is not None: self.use_tqdm = use_tqdm
        if threads is not None: self.threads = threads

        if "mp" not in globals():
            print("'multiprocess' module not available, parallel execution not possible! Falling back to single-thread execution.",file=sys.stderr)
            self.threads = 1


    def fourier_coeff_from_curve(self, curve_points : np.ndarray, coeff_count : int): 

        f_x, f_y = curve_points.T

        # Add a progress bar if tqdm is available
        if "tqdm" in globals() and self.use_tqdm:
            range_iterable = tqdm.trange(-coeff_count,coeff_count+1,self.batch_size)
        else:
            range_iterable = range(-coeff_count,coeff_count+1,self.batch_size)

        def _runner(bi):
            coeff_range = np.arange(bi,min(bi+self.batch_size,coeff_count+1))

            # Prepare trig function coeffs
            coscoeff = np.cos(-2*np.pi*np.linspace(0,1,len(curve_points),endpoint=False)[None,:]*coeff_range[:,None])
            sincoeff = np.sin(-2*np.pi*np.linspace(0,1,len(curve_points),endpoint=False)[None,:]*coeff_range[:,None])

            # This is analogous to cuadrature between 0 and 1
            batch_rc = np.mean(coscoeff*f_x[None] - sincoeff*f_y[None],axis=1) # Real Coefficients
            batch_ic = np.mean(coscoeff*f_y[None] + sincoeff*f_x[None],axis=1) # Imaginary Coefficients

            return batch_rc, batch_ic

        if self.threads == 1:
            rc, ic = zip(*map(_runner, range_iterable))
        else:
            with mp.Pool(self.threads) as p:
                rc, ic = zip(*p.map(_runner, range_iterable))

        # Concatenate all coefficients
        rc, ic = np.concat(rc), np.concat(ic)

        # Coefficient indices
        ci = np.arange(-coeff_count,coeff_count+1,dtype=int) 

        # $f_x(t) \simeq \sum_{i \in ci} cc(i)*cos(2*i*\pi*t)-sc(i)*sin(2*i*\pi*t) : t \in [0,1]$
        # $f_y(t) \simeq \sum_{i \in ci} sc(i)*cos(2*i*\pi*t)+cc(i)*sin(2*i*\pi*t) : t \in [0,1]$
        return ci, (rc, ic) 

    def evaluate_fourier_sum(self, parameter : np.ndarray, coeffs : np.ndarray, coeff_idx : np.ndarray = None):

        if coeff_idx is None:
            coeff_count = (len(coeffs)-1)//2
            coeff_idx = np.arange(-coeff_count,coeff_count+1,dtype=int)

        # Add a progress bar if tqdm is available
        if "tqdm" in globals() and self.use_tqdm:
            range_iterable = tqdm.trange(0,len(parameter),self.batch_size)
        else:
            range_iterable = range(0,len(parameter),self.batch_size)

        def _runner(bi):
            batch_parameter = parameter[bi:bi+self.batch_size]

            # Prepare trig function coeffs
            coscoeff = np.cos(2*np.pi*batch_parameter[None,:]*coeff_idx[:,None])
            sincoeff = np.sin(2*np.pi*batch_parameter[None,:]*coeff_idx[:,None])

            # Compute sum
            batch_X = np.sum(coscoeff*coeffs[0][:,None] - sincoeff*coeffs[1][:,None],axis=0)
            batch_Y = np.sum(sincoeff*coeffs[0][:,None] + coscoeff*coeffs[1][:,None],axis=0)

            return batch_X, batch_Y

        if self.threads == 1:
            X, Y = zip(*map(_runner, range_iterable))
        else:
            with mp.Pool(self.threads) as p:
                X, Y = zip(*p.map(_runner, range_iterable))

        X, Y = np.concat(X), np.concat(Y)
        return np.stack((X,Y),axis=1)

# Add a singleton
FourierRuntime.default_runtime = FourierRuntime()

# Define API functions
def fourier_coeff_from_curve(curve_points : np.ndarray, coeff_count : int):
    return FourierRuntime.default_runtime.fourier_coeff_from_curve(curve_points, coeff_count)

def evaluate_fourier_sum(parameter : np.ndarray, coeffs : np.ndarray, coeff_idx : np.ndarray = None):
    return FourierRuntime.default_runtime.evaluate_fourier_sum(parameter, coeffs, coeff_idx)
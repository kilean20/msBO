import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import io
import contextlib
import warnings

@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

@contextlib.contextmanager
def capture_warnings():
    warnings.simplefilter("always")
    yield
    with warnings.catch_warnings(record=True) as ws:
        for w in ws:
            print(w.message)
            print(f"{w.filename}, line {w.lineno}")


def defaultKeyVal(d,k,v):
  if k in d.keys():
    return d[k]
  else:
    return v
    

class dictClass(dict):
  """ 
  This class is essentially a subclass of dict
  with attribute accessors, one can see which attributes are available
  using the `keys()` method.
  """
  def __dir__(self):
      return self.keys()
    
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)
  if dict==None:
    __setattr__ = {}.__setitem__
    __delattr__ = {}.__delitem__
  else:
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

  def __repr__(self):
    if self.keys():
      L = list(self.keys())
      L = [str(L[i]) for i in range(len(L))]
      m = max(map(len, L)) + 1
      f = ''
      for k, v in self.items():
        if isinstance(v,dict):
          f = f + '\n'+str(k).rjust(m) + ': ' + repr(k) + ' class'
      return f
    else:
      return self.__class__.__name__ + "()"

  def find_key(self,val):
    if val==None :
      return
    for k in self.keys():
      if self[k]==val:
        return 
    
class timeout():
    def __init__(self,t):
        self.t0 = time.monotonic()
        self.t = t
    def __call__(self):
        if time.monotonic() - self.t0 > self.t:
            return True
        else:
            return False
    

class OptimizationTimeoutError(Exception):
    r"""Exception raised when optimization times out."""

    def __init__(
        self, /, *args: Any, current_x: np.ndarray, runtime: float
    ) -> None:
        r"""
        Args:
            current_x: A numpy array representing the current iterate.
            runtime: The total runtime in seconds after which the optimization
                timed out.
        """
        self.current_x = current_x
        self.runtime = runtime
        
        
class NotYetImplementedError(Exception):
    r"""Exception raised when optimization times out."""
    def __init__(self) -> None:
        pass


def unique_xy(x, y=None):
    _, indices, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    duplicate_indices = np.where(counts > 1)[0]
    duplicate_rows_indices = np.array([np.where(indices == i)[0] for i in duplicate_indices])
    
    if len(duplicate_rows_indices)>0:
        print("x data has duplicates. will remove duplicates and will average corresponding ys")
        uniq_x = []
        uniq_y = []
        for i,x_ in enumerate(x):
            if i in duplicate_rows_indices.flatten():
                for idups in duplicate_rows_indices:
                    if i==idups[0]:
                        uniq_x.append(x_)
                        if y is not None:
                            uniq_y.append(np.mean(y[idups]))
            else:
                uniq_x.append(x_)
                if y is not None:
                    uniq_y.append(y.flatten()[i])
        x = np.array(uniq_x)
        if y is not None:
            y = np.array(uniq_y)
    return x, y

    

def minimize_with_timeout(
    fun: Callable,
    x0: np.ndarray,
    method: Optional[str] = None,
    jac: Optional[Union[str, Callable, bool]] = None,
    hess: Optional[Union[str, Callable, optimize.HessianUpdateStrategy]] = None,
    hessp: Optional[Callable] = None,
    bounds: Optional[Union[Sequence[Tuple[float, float]], optimize.Bounds]] = None,
    constraints=(),  # Typing this properly is a s**t job
    tol: Optional[float] = 1e-3,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout_sec: Optional[float] = None,
) -> optimize.OptimizeResult:
    r"""Wrapper around scipy.optimize.minimize to support timeout.
    This method calls scipy.optimize.minimize with all arguments forwarded
    verbatim. The only difference is that if provided a `timeout_sec` argument,
    it will automatically stop the optimziation after the timeout is reached.
    Internally, this is achieved by automatically constructing a wrapper callback
    method that is injected to the scipy.optimize.minimize call and that keeps
    track of the runtime and the optimization variables at the current iteration.
    """
    if timeout_sec:

        start_time = time.monotonic()
        callback_data = {"num_iterations": 0}  # update from withing callback below

        def timeout_callback(xk: np.ndarray) -> bool:
            runtime = time.monotonic() - start_time
            callback_data["num_iterations"] += 1
            if runtime > timeout_sec:
                raise OptimizationTimeoutError(current_x=xk, runtime=runtime)
            return False

        if callback is None:
            wrapped_callback = timeout_callback

        elif callable(method):
            raise NotImplementedError(
                "Custom callable not supported for `method` argument."
            )

        elif method == "trust-constr":  # special signature

            def wrapped_callback(
                xk: np.ndarray, state: optimize.OptimizeResult
            ) -> bool:
                # order here is important to make sure base callback gets executed
                return callback(xk, state) or timeout_callback(xk=xk)

        else:

            def wrapped_callback(xk: np.ndarray) -> None:
                timeout_callback(xk=xk)
                callback(xk)

    else:
        wrapped_callback = callback

    try:
        return optimize.minimize(
            fun=fun,
            x0=x0,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=wrapped_callback,
            options=options,
        )
#         if jac is None:
#             jac = '2-point'
#         bounds = np.array(bounds)
#         return optimize.least_squares(
#             fun=fun,
#             x0=x0,
#             jac=jac,
#             bounds=bounds,
#             xtol=tol,
# #             options=options,
#         )
    except OptimizationTimeoutError as e:
        msg = f"Optimization timed out after {e.runtime} seconds."
        current_fun, *_ = fun(e.current_x, *args)

        return optimize.OptimizeResult(
            fun=current_fun,
            x=e.current_x,
            nit=callback_data["num_iterations"],
            success=False,  # same as when maxiter is reached
            status=1,  # same as when L-BFGS-B reaches maxiter
            message=msg,
        )

    
                
def progressbar(it, prefix="", size=40, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {int(j/count*100)}%/{100}%", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
    
    
    
def minimize_with_restarts(
    fun, 
    bounds, 
    x0 = None,
    num_restarts=100, 
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    constraints=None, 
    tol=None,
    callback=None,
    options=None,
    timeout_sec=None,
    optmization_stopping_criteria=None,
    ):

    
    if num_restarts is None:
        num_restarts=1       
    if num_restarts<1:
        num_restarts=1
    
    start_time = time.monotonic()
        
    bounds = np.array(bounds)
    xdim = len(bounds)
    
    candidates = {'success':{'x':[],'fun':[]},
                  'fail':{'x':[],'fun':[]}}
    for i_restart in range(num_restarts):
        if len(candidates['success']['fun'])>2 and optmization_stopping_criteria is not None:
            if np.any([stop() for stop in optmization_stopping_criteria]):
                break
            
        if timeout_sec is not None:
            timeout_sec_ = timeout_sec - (time.monotonic() - start_time)
        else:
            timeout_sec_ = None
        
        if x0 is not None and i_restart==0:
            x0 = x0
        else:
            x0 = np.random.rand(xdim)*(bounds[:,1]-bounds[:,0])+bounds[:,0]
        result = minimize_with_timeout(
                    fun,
                    x0,
                    method,
                    jac,
                    hess,
                    hessp,
                    bounds,
                    constraints, 
                    tol,
                    callback,
                    options,
                    timeout_sec_,)
        
        if result.success:
            if type(result.x) is not np.ndarray:
                print('result.x',result.x)
            assert type(result.x) == np.ndarray
            if type(result.fun) is not float:
                assert result.fun.shape==(1,)
                result.fun = float(result.fun[0])
            else:
                assert type(result.fun) == float
            candidates['success']['x'].append(result.x)
            candidates['success']['fun'].append(result.fun)
        else:
            if type(result.x) == np.ndarray and type(result.fun) == float:
                candidates['fail']['x'].append(result.x)
                candidates['fail']['fun'].append(result.fun)               
    
    success = True
    if len(candidates['success']['fun'])>=1:
        i = np.argmin(candidates['success']['fun'])
        x = candidates['success']['x'][i]
        fun = candidates['success']['fun'][i]
    elif len(candidates['fail']['fun'])>=1:
        i = np.argmin(candidates['fail']['fun'])
        x = candidates['fail']['x'][i]
        fun = candidates['fail']['fun'][i]
        success = False
    else:
        success = False
        return optimize.OptimizeResult(
            time=time.monotonic()-start_time,
            success=success)
    
    return optimize.OptimizeResult(
            fun=fun,
            x=x,
            time=time.monotonic()-start_time,
            success=success)


def maximize_with_restarts(
    fun, 
    bounds, 
    x0 = None,
    num_restarts=100, 
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    constraints=None, 
    tol=None,
    callback=None,
    options=None,
    timeout_sec=None,
    optmization_stopping_criteria=None):
    
    
    def neg_fun(x):
        return -fun(x)
    
    if jac is not None:
        def neg_jac(x):
            return -jac(x)
    else:
        neg_jac = None
    
    return minimize_with_restarts(
        neg_fun, 
        bounds, 
        x0,
        num_restarts, 
        method, 
        neg_jac, 
        hess, 
        hessp, 
        constraints, 
        tol, 
        callback, 
        options,
        timeout_sec,
        optmization_stopping_criteria)



def plot_2D_projection(
                        func,
                        bounds,
                        dim_xaxis = 0,
                        dim_yaxis = 1,
                        grid_ponits_each_dim = 25, 
                        project_maximum = True,
                        project_mode = None,
                        fixed_values_for_each_dim = None, 
                        overdrive = False,
                        fig = None,
                        ax = None,
                        colorbar = True,
                        dtype = np.float32 ):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        
        n_fixed = 0
        if fixed_values_for_each_dim is not None:
            n_fixed = len(fixed_values_for_each_dim.keys())
            assert dim_xaxis not in fixed_values_for_each_dim.keys()
            assert dim_yaxis not in fixed_values_for_each_dim.keys()
        else:
            fixed_values_for_each_dim = {}
            
        dim = len(bounds)  
        
        project_minimum = False
        project_mean = False
        if project_mode is not None:
            if project_mode in ['MAX', 'max', 'Max']:
                project_maximum = True
            elif project_mode in ['MIN', 'min', 'Min']:
                project_minimum = True
                project_maximum = False
            elif project_mode in ['MEAN', 'mean', 'Mean','Average', 'Ave', 'ave']:
                project_mean = True,
                project_maximum = False
        if dim > 2+n_fixed:
            assert project_minimum + project_maximum + project_mean == 1
        
        batch_size = 1
        for n in range(dim-2-n_fixed):
            batch_size*=grid_ponits_each_dim
            if batch_size*dim > 2e4:
                if overdrive or not (project_minimum or project_mean):
                    batch_size = int(batch_size/grid_ponits_each_dim)
                    print("starting projection plot...")
                    break
                else:
                    raise RuntimeError("Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_ponits_each_dim' or turn on 'overdrive' if long time waiting is OK'")
        n_batch = int(grid_ponits_each_dim**(dim-n_fixed-2)/batch_size)
        linegrid = np.linspace(0,1,grid_ponits_each_dim)
        x_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim,dim))
        y_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim))
        
        n = 0
        for i in progressbar(range(grid_ponits_each_dim)):
            bounds_xaxis = bounds[dim_xaxis,:]
            for j in range(grid_ponits_each_dim):
                bounds_yaxis = bounds[dim_yaxis,:]
                x_grid[n,dim_xaxis] = linegrid[i]*(bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n,dim_yaxis] = linegrid[j]*(bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_maximum or project_mean) and dim > 2+n_fixed:
                    inner_grid = []
                    for d in range(dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n,dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n,dim_yaxis]])
                        elif d in fixed_values_for_each_dim.keys():
                            inner_grid.append([fixed_values_for_each_dim[d]])
                        else:
                            inner_grid.append(np.linspace(bounds[d,0],
                                                          bounds[d,1],
                                                          grid_ponits_each_dim))
                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat) for x in inner_grid),dtype=dtype).T
                    
                    y = []
                    for b in range(n_batch):
                        i1 = b*batch_size
                        i2 = i1 + batch_size
                        x_batch = inner_grid[i1:i2]
                        y.append(func(x_batch))
                    y = np.concatenate(y, axis=0)

                    if project_minimum or project_maximum:
                        if project_minimum:
                            arg_project = np.nanargmin
                        elif project_maximum:
                            arg_project = np.nanargmax
                        iproj = arg_project(y)
                        y_grid[n] = y[iproj]
                        x_grid[n,:] = inner_grid[iproj]
                    elif project_mean:
                        y_grid[n] = np.nanmean(y_mean)
                n+=1
                
        if dim==2+n_fixed:
            if fixed_values_for_each_dim is not None:
                for dim,val in fixed_values_for_each_dim.items():
                    x_grid[:,dim]  = val   
            y_grid  = func(x_grid)
            
        iNaN = np.any([np.isnan(x_grid).any(axis=1),np.isnan(y_grid)],axis=0)
        x_grid = x_grid[~iNaN,:]
        y_grid = y_grid[~iNaN  ]
                           
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3),dpi=96)
        cs = ax.tricontourf(x_grid[:,dim_xaxis], x_grid[:,dim_yaxis], y_grid, levels=64, cmap="viridis");
        if colorbar:
            fig.colorbar(cs,ax=ax,shrink=0.95)
            
        return cs

            
# def _init_population_lhs(nsample, ndim):
#     """
#     Initializes the population with Latin Hypercube Sampling.
#     Latin Hypercube Sampling ensures that each parameter is uniformly
#     sampled over its range.
#     rng : random generator
#     """

#     segsize = 1.0 / nsample
#     samples = (segsize * np.random.uniform(size=(nsample,ndim))
#     # Offset each segment to cover the entire parameter range [0, 1)
#                + np.linspace(0., 1., nsample,
#                              endpoint=False)[:, np.newaxis])
#     population = np.zeros_like(samples)

#     # Initialize population of candidate solutions by permutation of the
#     # random samples.
#     for j in range(ndim):
#         order = np.random.permutation(range(nsample))
#         population[:, j] = samples[order, j]

#     return population


def _rightmost_zero(n):
    """Position of the lowest 0-bit in the binary representation of integer `n`."""
    s = np.binary_repr(n)
    i = s[::-1].find("0")
    if i == -1:
        i = len(s)
    return i


def _generator(dimension, skip=0):
    """Generator for the Sobol sequence"""
    DIMS = 23  # maximum number of dimensions
    BITS = 30  # maximum number of bits

    if not (1 <= dimension <= DIMS):
        raise ValueError("Sobol: dimension must be between 1 and %i." % DIMS)

    # initialize direction numbers
    V = np.zeros((DIMS, BITS), dtype=int)
    data = np.array([
                        [7,1,1,0,0,0,0,0,0,0,0,0,0,0],
                        [11,1,3,7,0,0,0,0,0,0,0,0,0,0],
                        [13,1,1,5,0,0,0,0,0,0,0,0,0,0],
                        [19,1,3,1,1,0,0,0,0,0,0,0,0,0],
                        [25,1,1,3,7,0,0,0,0,0,0,0,0,0],
                        [37,1,3,3,9,9,0,0,0,0,0,0,0,0],
                        [59,1,3,7,13,3,0,0,0,0,0,0,0,0],
                        [47,1,1,5,11,27,0,0,0,0,0,0,0,0],
                        [61,1,3,5,1,15,0,0,0,0,0,0,0,0],
                        [55,1,1,7,3,29,0,0,0,0,0,0,0,0],
                        [41,1,3,7,7,21,0,0,0,0,0,0,0,0],
                        [67,1,1,1,9,23,37,0,0,0,0,0,0,0],
                        [97,1,3,3,5,19,33,0,0,0,0,0,0,0],
                        [91,1,1,3,13,11,7,0,0,0,0,0,0,0],
                        [109,1,1,7,13,25,5,0,0,0,0,0,0,0],
                        [103,1,3,5,11,7,11,0,0,0,0,0,0,0],
                        [115,1,1,1,3,13,39,0,0,0,0,0,0,0],
                        [131,1,3,1,15,17,63,13,0,0,0,0,0,0],
                        [193,1,1,5,5,1,59,33,0,0,0,0,0,0],
                        [137,1,3,3,3,25,17,115,0,0,0,0,0,0],
                        [145,1,1,7,15,29,15,41,0,0,0,0,0,0],
                        [143,1,3,1,7,3,23,79,0,0,0,0,0,0],
                        [241,1,3,7,9,31,29,17,0,0,0,0,0,0],
                    ], dtype=int)
    poly = data[:, 0]
    V[:, :13] = data[:, 1:14]
    V[0, :] = 1
    for i in range(1, dimension):
        m = len(np.binary_repr(poly[i])) - 1
        include = np.array([int(b) for b in np.binary_repr(poly[i])[1:]])
        for j in range(m, BITS):
            V[i, j] = V[i, j - m]
            for k in range(m):
                if include[k]:
                    V[i, j] = np.bitwise_xor(V[i, j], 2 ** (k + 1) * V[i, j - k - 1])
    V = V[:dimension] * 2 ** np.arange(BITS)[::-1]

    point = np.zeros(dimension, dtype=int)

    # fast-forward
    for i in range(skip):
        point = np.bitwise_xor(point, V[:, _rightmost_zero(i)])

    # start sampling
    for i in range(skip, 2 ** BITS):
        point = np.bitwise_xor(point, V[:, _rightmost_zero(i)])
        yield point / 2 ** BITS


def _get_sobol_sample(n_points, dimension,skip=0):
    """Generate a Sobol point set.
    Parameters
    ----------
    dimension : int
        Number of dimensions
    n_points : int, optional
        Number of points to sample
    skip : int, optional
        Number of points in the sequence to skip, by default 0
    Returns
    -------
    array, shape=(n_points, dimension)
        Samples from the Sobol sequence.
    """
    sobol = _generator(dimension, skip)
    points = np.empty((n_points, dimension))
    for i in range(n_points):
        points[i] = next(sobol)
    return points


def _init_population_qmc(n, d, qmc_engine='sobol',seed=None):
    """Initializes the population with a QMC method.
    QMC methods ensures that each parameter is uniformly
    sampled over its range.
    Parameters
    ----------
    qmc_engine : str
        The QMC method to use for initialization. Can be one of
        ``latinhypercube``, ``sobol`` or ``halton``.
    """
    try:
        from scipy.stats import qmc
        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=d, seed=seed)
        elif qmc_engine == 'sobol':        
            sampler = qmc.Sobol(d=d, seed=seed)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=d, seed=seed)
        else:
            raise ValueError("qmc_engine",qmc_engine,"is not recognized.")
        return sampler.random(n=n)
    except:
        print("scipy version mismatch. 'scipy.stat.qmc' is not imported. Using custom halton seqeunce instead")
        return _get_sobol_sample(n,d)
                  

def proximal_ordered_init_sampler(n,
                                  bounds,
                                  x0,
                                  ramping_rate=None,
                                  polarity_change_time=None,
                                  method='sobol',seed=None):
    if n<1:
        raise ValueError('n must be larger than 0') 
    bounds = np.array(bounds,dtype=np.float64)
    d = len(bounds)
    x0 = np.atleast_2d(x0)
    _,xd = x0.shape
    assert _==1
    assert xd==d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        samples = list(_init_population_qmc(n, d, method, seed) * 
                       (bounds[:, 1] - bounds[:, 0])[None, :] + bounds[:, 0][None, :])
    return order_samples( samples,
                          x0=x0,
                          ramping_rate=ramping_rate,
                          polarity_change_time=polarity_change_time)

def order_samples(samples,
                  x0,
                  ramping_rate=None,
                  polarity_change_time=None):
    
    ordered_samples = []
    x0 = np.atleast_2d(x0)
    _,d = x0.shape
    assert _ == 1
    samples = np.atleast_2d(samples)
    n = len(samples)
    if n == 1:
        return samples
    samples = samples.tolist()
    if ramping_rate is None:
        ramping_rate = np.ones(d)*5
    polarity_change_time = polarity_change_time or 15
    
    while len(ordered_samples)<n:
        x = np.array(samples)
        sign_x  = np.sign(x)
        sign_x0 = np.sign(x0)
        is_not_zero = np.logical_and(sign_x != 0, sign_x0 != 0)
        is_pol_crossing = (sign_x != sign_x0)
        distance = polarity_change_time * np.all(np.logical_and(is_pol_crossing, is_not_zero), axis=1).astype(np.float64)
        distance += np.max( np.abs(x-x0)/ramping_rate, axis=1)
        ordered_samples.append(samples.pop(np.argmin(distance)))
        x0 = ordered_samples[-1]
        
    return np.array(ordered_samples)
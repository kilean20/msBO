from typing import List, Union, Optional, Callable
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

torch.set_num_threads(4)

import botorch
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement,qUpperConfidenceBound,qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import MCAcquisitionObjective, GenericMCObjective

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import settings
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.means import LinearMean
from gpytorch.constraints import GreaterThan

from .utils import proximal_ordered_init_sampler, dictClass
from .construct_machineIO import MultiStateEvaluator
#import MultiStateEvaluator


def MultiStateObject_2_MultiTaskObject(multi_state_objective_fn, n_state, n_obj):
    '''
    multi_state_objective_fn: should take tensor of shape (n_data, n_state, n_obj)
    it returns multi_task_objective_fn what should take tensor of shape (n_data, n_task = n_obj*n_state)
    '''
    # print("n_state, n_obj",n_state, n_obj)
    def multi_task_objective_fn(mty, X=None):
        # print("mty.shape",mty.shape)
        if mty.shape[-1] == 1:
            shape = list(mty.shape[:-1]) + [1]
            # print("shape",shape)
            shape[-2] = int(mty.shape[-2]/(n_state*n_obj))
            # print("shape",shape)
        else:
            shape = mty.shape[:-1]
        
        # n_obj = int(n_task/n_state)
        #assert n_state*n_obj == n_task #debug
        #msy = np.zeros(n_data,n_state,n_obj)
        #for i_state in range(n_state):
        #    msy[:,i,:] = mty[:,i_state*n_obj:(i_state+1)*n_obj]
        obj = multi_state_objective_fn(mty.view(-1,n_state,n_obj))
        # print("obj.shape",obj.shape)
        obj = obj.view(*shape)
        # print("obj.shape",obj.shape)
        return obj
    return multi_task_objective_fn


def ldata_2_mtGPdata(l_train_X, l_train_Y=None):
    l_train_X_taskLabel = []
    for i, x in enumerate(l_train_X):
        if x is not None:
            l_train_X_taskLabel.append(torch.cat([x, i * torch.ones(x.shape[0], 1)], dim=1))
    train_X = torch.cat(l_train_X_taskLabel, dim=0)

    if l_train_Y is None:
        return train_X
    else:
        # for y in l_train_Y : #debug
        #     if y is not None:
        #         print("y.shape",y.shape)
        l_train_Y_filtered = [y for y in l_train_Y if y is not None]  # y should be None or tensor of shape (n_data,1)
        train_Y = torch.cat(l_train_Y_filtered, dim=0)
        return train_X, train_Y

def ldata_2_MultiStateGPdata(l_train_X, n_obj, l_train_Y=None):
    l_train_X_expanded = []
    # print("l_train_X",l_train_X)
    for x in l_train_X:
        l_train_X_expanded += [x]*n_obj
    if l_train_Y is None:
        l_train_Y_expanded = None
    else:
        l_train_Y_expanded = []
        for y in l_train_Y: 
            if y is None:
                l_train_Y_expanded += [None]*n_obj
            else:
                l_train_Y_expanded += [y[:,i:i+1] for i in range(n_obj)]   
    return ldata_2_mtGPdata(l_train_X_expanded, l_train_Y_expanded)


        
class MultiStateBO:
    def __init__(self,
        machineIO,
        objective_PVs: List[str],
        control_CSETs: List[str],
        control_RDs  : List[str],
        control_tols : Union[List[float], np.ndarray],
        control_min  : Union[List[float], np.ndarray],
        control_max  : Union[List[float], np.ndarray],
        state_CSETs  : List[str],
        state_RDs    : List[str],
        state_tols   : List[str],
        state_vals   : Union[List[float], np.ndarray],
        state_names  : Optional[List[str]] = None,
        monitor_PVs  : Optional[List[str]] = None,
        control_couplings: Optional[dict] = None,
        multi_state_objective_fn: Optional[Callable] = None,
        local_optimization : Optional[bool] = False,
        local_bound_size   : Optional[Union[List[float], np.ndarray]] = None,
        ):
        """
        multi_state_objective_fn: should take tensor of shape (n_data, n_state, n_obj)
        state_vals should be shape of (n_state, n_state_CSETs)
        """
        machineIO.clear_history()
        self.machineIO = machineIO
        self.n_control = len(control_CSETs)
        self.n_state = len(state_vals)
        self.n_obj  = len(objective_PVs)
        self.objective_PVs = objective_PVs
        self.control_CSETs = control_CSETs
        self.control_RDs   = control_RDs
        self.control_tols  = control_tols
        self.control_min   = np.array(control_min)
        self.control_max   = np.array(control_max)
        self.state_CSETs   = state_CSETs
        self.state_RDs     = state_RDs
        self.state_tols    = state_tols
        self.state_vals    = state_vals
        self.state_names   = state_names
        self.monitor_PVs   = monitor_PVs
        self.local_optimization = local_optimization

        if local_optimization:
            if local_bound_size is None:
                self.local_bound_size = 0.1*(control_max - control_min)
            else: 
                self.local_bound_size = local_bound_size
        if monitor_PVs is None:
            monitor_PVs = []
        
        self.evaluator = MultiStateEvaluator(
            machineIO     = machineIO,
            control_CSETs = control_CSETs,
            control_RDs   = control_RDs,
            control_tols  = control_tols,
            state_CSETs   = state_CSETs,
            state_RDs     = state_RDs,
            state_tols    = state_tols,
            state_vals    = state_vals,
            state_names   = state_names,
            monitor_PVs   = monitor_PVs + objective_PVs,
            control_couplings = control_couplings,
            )
        x0, df = self.machineIO.fetch_data(self.control_CSETs,time_span=0.2)
        # self.machineIO.clear_history()
        self.x0 = np.array(x0)
        assert self.x0.shape == (self.n_control,)
        
        self.timecost = {'query':[],'fit':[]}

        if multi_state_objective_fn is not None:
            self.init_objective(multi_state_objective_fn)
        else:
            logging.warning("multi_state_objective_fn is not provided, must be initialized with init_objective before calling init")

    def init_objective(self, multi_state_objective_fn):
        self.multi_state_objective_fn = multi_state_objective_fn
        self.multi_task_objective_fn = MultiStateObject_2_MultiTaskObject(multi_state_objective_fn, self.n_state, self.n_obj)
        self.mc_objective  = GenericMCObjective(self.multi_task_objective_fn)

    def get_state_index(self, state_names):
        return [self.state_names.index(state_name) for state_name in state_names]
    def get_objective_index(self, objective_PVs):
        return [self.objective_PVs.index(objective_PV) for objective_PV in objective_PVs]

    @property
    def control_bounds(self):
        return np.array(list(zip(self.control_min, self.control_max)))
        
    @property
    def control_bounds_botorch(self):
        return torch.tensor(np.array([self.control_min,self.control_max]))

    def init(self,
             n_init,
             init_bound_size = None):
        if not hasattr(self,"multi_state_objective_fn"):
            raise ValueError("multi_state_objective_fn is not initialized, call init_objective before init")

        if self.local_optimization:        
            lower_bound = np.maximum(self.x0 -0.5*self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self.x0 +0.5*self.local_bound_size, self.control_max)
            bounds = np.vstack((lower_bound, upper_bound)).T  # Shape bounds as (n, 2)
        else:
            bounds = self.control_bounds

        init_x = proximal_ordered_init_sampler(
            n_init-1,
            bounds=bounds,
            x0=self.x0,
            ramping_rate=None,
            polarity_change_time=None, 
            method='sobol',
            seed=None)
        init_x = np.vstack((self.x0.reshape(1,-1),init_x))
        
        l_train_X = [[] for _ in range(self.n_state)]
        l_train_Y = [[] for _ in range(self.n_state)]

        for i_state, state_name in enumerate(self.evaluator.state_names):
            for isample,x in enumerate(init_x):
                future = self.evaluator.submit(x,i_state)
                df, ramping_df = self.evaluator.get_result(future)
                l_train_X[i_state].append(df[self.control_RDs].mean().values)
                l_train_Y[i_state].append(df[self.objective_PVs].mean().values)
                # display(df) #debug
        t0 = time.monotonic()
        self.msGP = msGP(l_train_X = l_train_X, l_train_Y = l_train_Y)
        self.timecost['fit'].append(time.monotonic()-t0)

    def query_candidate(self,bounds,acq_function=None,beta=9):
        t0 = time.monotonic()
        if acq_function is None:
            acq_function = qUpperConfidenceBound(model=self.msGP.model,
                                            beta = beta,
                                            objective=self.mc_objective,         
                                            )
        
        # Optimize acquisition function to get next candidate (q=1 for single candidate)
        candidate, _ = optimize_acqf( 
            acq_function=acq_function,
            bounds=bounds,   # Define your control space bounds
            q=1,             # Number of candidates to generate
            num_restarts= 4,  # Number of restarts for optimization
            raw_samples = 16,   # Number of raw samples for initial candidate set
            # timeout_sec = self.machineIO.fetch_data_time_span + self.machineIO.ensure_set_timewait_after_ramp,
        )
        self.timecost['query'].append(time.monotonic()-t0)
        return candidate


    def optimize_global(self, neval=3, i_state=None, beta=9):
        if i_state is None:
            i_states = range(self.n_state)
        else:
            i_states = [i_state]

        for i_state in i_states:
            for ieval in range(neval):
                acq_function = self.get_acq_function(i_state = i_state, beta=beta)
                candidate = self.query_candidate(self.control_bounds_botorch, acq_function)
                future = self.evaluator.submit(candidate.view(-1).detach().cpu().numpy(), i_state)
                df, ramping_df = self.evaluator.get_result(future)
                x = df[self.control_RDs].mean().values
                y = df[self.objective_PVs].mean().values
                self.msGP.add_new_data(i_state, torch.tensor(x).view(1,-1), torch.tensor(y).view(1,-1))
                t0 = time.monotonic()
                self.msGP.fit()
                self.timecost['fit'].append(time.monotonic()-t0)

    
    def get_acq_function(self,i_state=None,beta=9,X_pending=None):
        if beta < 0.1:
            X_pending = None
        else:
            if X_pending is not None:
                X_pending = torch.atleast_2d(X_pending)
                assert X_pending.shape[1] == self.n_control
        
            if i_state is not None:
                # acq_function = qNoisyExpectedImprovement(
                #                                     model=self.msGP.model,
                #                                     objective=self.mc_objective,
                #                                     X_baseline= self.msGP.l_train_X[i_state],#self.msGP.train_X,
                #                                     )
                
                if X_pending is None:
                    X_pending = self.msGP.l_train_X[i_state]
                else:
                    print("X_pending.shape",X_pending.shape)
                    X_pending = torch.stack((X_pending,self.msGP.l_train_X[i_state]))
                    print("X_pending.shape",X_pending.shape)
            
        acq_function = qUpperConfidenceBound(model=self.msGP.model,
                                            beta = beta,
                                            objective=self.mc_objective,
                                            X_pending = X_pending
                                            )
            
        return acq_function
            
            

class mtGP:
    def __init__(self, l_train_X, l_train_Y, GP_kwargs=None, multi_task_objective_fn=None):
        """
        Args:
            l_train_X (list of torch.Tensor): List of length n_task. Each element is  input for each task and should be tensor of shape (n_data, input_dim)
            l_train_Y (list of torch.Tensor): List of length n_task. Each element is output for each task and should be tensor of shape (n_data, 1) 
        """
        assert len(l_train_X) == len(l_train_Y)
        assert l_train_Y[0].shape[1] == 1
        self.n_task = len(l_train_X)
        self.input_dim = l_train_X[0].shape[1]
        
        self.l_train_X = l_train_X
        self.l_train_Y = l_train_Y
        if GP_kwargs is None:
            GP_kwargs = {}
        else:
            GP_kwargs.pop('task_feature', None)
            GP_kwargs.pop('input_transform', None)
            GP_kwargs.pop('outcome_transform', None)
        self.GP_kwargs = GP_kwargs       

        self.multi_task_objective_fn = multi_task_objective_fn
        
        # self.mean_module = LinearMean(input_size=l_train_X[0].shape[1])
        self.train_X, self.train_Y = ldata_2_mtGPdata(self.l_train_X,self.l_train_Y)
        self.fit()
        

    def add_new_data(self, i_task, candidate_x, evaluated_y):
        """
        Adds new data for a specific task, updates the internal storage, and re-combines the data.
        
        Args:
            i_task (int): Index of the task to which new data belongs.
            candidate_x (torch.Tensor): New input data for the task.
            evaluated_y (torch.Tensor): New output data for the task.
        """
        assert i_task < self.n_task, "Invalid task index."
        self.l_train_X[i_task] = torch.cat([self.l_train_X[i_task], candidate_x], dim=0)
        self.l_train_Y[i_task] = torch.cat([self.l_train_Y[i_task], evaluated_y], dim=0)
        
        # Recombine data after adding new samples
        self.train_X, self.train_Y = ldata_2_mtGPdata(self.l_train_X,self.l_train_Y)

    def fit(self,lr=0.2,epochs=200):
        model = MultiTaskGP(
            self.train_X, 
            self.train_Y, 
            task_feature=-1,  # The last column indicates the task index
            input_transform=botorch.models.transforms.input.Normalize(self.train_X.shape[1] - 1, 
                                                                      indices=list(range(self.train_X.shape[1] - 1))
                                                                      ),
            outcome_transform=botorch.models.transforms.outcome.Standardize(1),
            **self.GP_kwargs,
            )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        # fit_gpytorch_mll(mll, options={"ftol": 1e-6, "maxiter": 200})
        opt = torch.optim.Adam(mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                        max_lr=lr,
                                                        div_factor=2.5,
                                                        pct_start=0.1, 
                                                        final_div_factor=5,
                                                        epochs=epochs, steps_per_epoch=1)
        if not hasattr(self,"train_history"):
            self.train_history = []
        self.train_history.append([])
        
        for _ in range(epochs):  # Number of iterations
            opt.zero_grad()
            model_output = model(*model.train_inputs)
            log_likelihood = mll(
                model_output,
                model.train_targets,
                *(model.transform_inputs(X=t_in) for t_in in model.train_inputs),
            )
            loss = -log_likelihood.mean()
            loss.backward()
            opt.step()
            scheduler.step()
            self.train_history[-1].append(loss.item())

        model.eval()
        self.model = model
        
        
    def predict(self, x, n_sample=None, multi_task_objective_fn=None):
        '''
        x : tensor, shape of n_data, input_dim
        '''
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x)
        x_taskLabel = [x]*self.n_task
        x_taskLabel = ldata_2_mtGPdata(x_taskLabel)
        posterior = self.model.posterior(x_taskLabel)
        #posterior.mean is shape of (len(x)*n_task,)
        pred_Y_mean = posterior.mean
        pred_Y_std  = posterior.stddev
        
        result = dictClass()
        result.mean = pred_Y_mean.view(self.n_task,len(x)).transpose(0, 1)
        result.std  = pred_Y_std .view(self.n_task,len(x)).transpose(0, 1)
        
        if n_sample:
            pred_Y_samples = posterior.rsample(torch.Size([n_sample])) #shape of (n_sample,len(x)*n_task)

            result.samples = pred_Y_samples.view(n_sample,self.n_task,len(x)).transpose(1, 2)
            
        multi_task_objective_fn = multi_task_objective_fn or self.multi_task_objective_fn
        if multi_task_objective_fn:  
            # mc_objective = GenericMCObjective(multi_task_objective_fn)
            result.obj_mean = multi_task_objective_fn(result.mean)
            if n_sample:
                result.obj_samples = multi_task_objective_fn(result.samples)  # shape of (n_sample, len(x))
        return result
        

class msGP(mtGP):
    def __init__(self, l_train_X, l_train_Y, GP_kwargs=None):
        """
        Args:
            l_train_X (list of torch.Tensor): List of length n_state. Each element is control_data for each n_state and should be tensor of shape (n_data, input_dim)
            l_train_Y (list of torch.Tensor): List of length n_state. Each element is objective_data for each n_state and should be tensor of shape (n_data, output_dim) 
        """
        assert len(l_train_X) == len(l_train_Y)
        self.n_state = len(l_train_X)
        for i,x in enumerate(l_train_X):
            if x is not None and not isinstance(x,torch.Tensor):
                l_train_X[i] = torch.tensor(np.array(x))
        for i,y in enumerate(l_train_Y):
            if x is not None and not isinstance(y,torch.Tensor):
                l_train_Y[i] = torch.tensor(np.array(y))
        self.n_obj   = l_train_Y[0].shape[1]
        self.n_task  = self.n_state*self.n_obj
        self.input_dim = l_train_X[0].shape[1]
        
        self.l_train_X = l_train_X
        self.l_train_Y = l_train_Y
        if GP_kwargs is None:
            GP_kwargs = {}
        else:
            GP_kwargs.pop('task_feature', None)
            GP_kwargs.pop('input_transform', None)
            GP_kwargs.pop('outcome_transform', None)
        self.GP_kwargs = GP_kwargs

        # self.mean_module = LinearMean(input_size=l_train_X[0].shape[1])
        self.train_X, self.train_Y = ldata_2_MultiStateGPdata(self.l_train_X, self.n_obj, self.l_train_Y)
        self.fit()
        
    def add_new_data(self, i_state, candidate_x, evaluated_y):
        """
        Adds new data for a specific task, updates the internal storage, and re-combines the data.
        
        Args:
            i_task (int): Index of the task to which new data belongs.
            candidate_x (torch.Tensor): New input data for the task.
            evaluated_y (torch.Tensor): New output data for the task.
        """
        assert i_state < self.n_state, "Invalid state index."
        self.l_train_X[i_state] = torch.cat([self.l_train_X[i_state], candidate_x], dim=0)
        self.l_train_Y[i_state] = torch.cat([self.l_train_Y[i_state], evaluated_y], dim=0)
        
        # Recombine data after adding new samples
        self.train_X, self.train_Y = ldata_2_MultiStateGPdata(self.l_train_X, self.n_obj, self.l_train_Y)
    
    def predict(self, x, n_sample=None, multi_state_objective_fn=None):
        '''
        x : tensor, shape of n_data, input_dim
        '''
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x)
        x_taskLabel = [x]*self.n_task
        x_taskLabel = ldata_2_mtGPdata(x_taskLabel)
        # print("x",x)
        # print("x_taskLabel",x_taskLabel)
        posterior = self.model.posterior(x_taskLabel)
        #posterior.mean is shape of (len(x)*n_task,)
        pred_Y_mean = posterior.mean
        pred_Y_std  = posterior.stddev

        # print("pred_Y_mean.shape",pred_Y_mean.shape)
        
        result = dictClass()
        result.mean = pred_Y_mean.view(self.n_task,len(x)).transpose(0,1).view(len(x),self.n_state,self.n_obj)
        # display(pred_Y_mean)
        # display(result.mean)
        result.std  = pred_Y_std .view(self.n_task,len(x)).transpose(0,1).view(len(x),self.n_state,self.n_obj)
        
        if n_sample:
            pred_Y_samples = posterior.rsample(torch.Size([n_sample])) #shape of (n_sample,len(x)*n_task)
            result.samples = pred_Y_samples.view(n_sample,self.n_task,len(x)).transpose(1,2).view(n_sample,len(x),self.n_state,self.n_obj)
        if multi_state_objective_fn:  
            result.obj_mean = multi_state_objective_fn(result.mean) # shape of (n_sample)
            # print("result.obj_mean.shape",result.obj_mean.shape)
            # print("self.n_state",self.n_state)
            # print("pred_Y_samples.shape",pred_Y_samples.shape)
            # multi_task_objective_fn = MultiStateObject_2_MultiTaskObject(multi_state_objective_fn,self.n_state,self.n_obj)
            # mc_objective = GenericMCObjective(multi_task_objective_fn)
            if n_sample:
                result.obj_samples = multi_state_objective_fn(result.samples)  # shape of (n_sample, len(x))
        return result
    
    

def find_common_rows(l_data, tols):
    """
    Find rows that are common across multiple 2D arrays or tensors within a specified tolerance.

    Parameters:
    l_data : list of np.ndarray or torch.Tensor
        A list of 2D arrays or tensors of shape (n_i, d), where n_i can be different for each array/tensor.
    tols : list of floats or np.ndarray
        A 1D array or list specifying the tolerance for each dimension.

    Returns:
    all_common_rows : np.ndarray
        An array of the common rows found across all arrays/tensors.
    all_close_indices : np.ndarray
        An array of the indices of the common rows in each respective array/tensor.
    """
    # Check if l_data contains torch tensors or numpy arrays
    is_tensor = isinstance(l_data[0], torch.Tensor)
    
    # Convert tolerance to numpy array for flexibility
    tols = np.array(tols).reshape(1, -1)
    
    # Get the dimension of the data
    dim = l_data[0].shape[1]
    assert tols.shape[1] == dim, "The tolerance dimension does not match data dimension!"
    
    # Add a small value to tolerance to avoid numerical precision issues
    tols = tols + 1e-12

    all_close_indices = []
    all_common_rows = []

    # Iterate through each row of the first array/tensor as the reference row
    for i, ref in enumerate(l_data[0]):
        ref = ref.unsqueeze(0) if isinstance(ref,torch.tensor) else ref[None, :]  # Adjust shape for broadcasting
        close_indices = [i]
        common_rows = [ref]

        # Check this row against all other arrays/tensors
        for test_array in l_data[1:]:
            test_array = test_array if not is_tensor else test_array.detach().cpu().numpy()  # Convert to numpy for easy calculations
            
            # Compute normalized differences between the reference row and all rows in the test array
            norm = np.mean(np.abs(ref - test_array) / tols, axis=1)
            
            # Find the index of the closest row
            argmin = norm.argmin()

            # Check if the closest row is within the tolerance threshold
            if norm[argmin] <= 1 / dim**0.5:
                close_indices.append(argmin)
                common_rows.append(test_array[argmin])
            else:
                break

        # If the row has a match in all arrays/tensors, record the indices and mean of common rows
        if len(close_indices) == len(l_data):  # Must match with all arrays/tensors
            all_close_indices.append(close_indices)
            all_common_rows.append(np.mean(np.array(common_rows), axis=0))

    return np.array(all_common_rows), np.array(all_close_indices)
    


def plot_msGP_over_states(
    msGP,
    control_values,
    state_names,
    objective_PVs,
    control_data_labels=None,
    objective_vals=None,
    multi_state_objective_fn=None,
    CL=1.96):
    '''
    plots:
        for each obj : xaxis: state,  yaxis: each_obj
        for total obj: xaxis: control label,  yaxis: total_obj
    control_values: array of shape (n_data,n_control)
    control_data_labels: list of str of length n_data
    state_names : list of str of length n_state
    objective_PVs : list of str of length n_obj
    objective_vals: array of shape (n_data, n_state, n_obj)
    '''
    with torch.no_grad():
        control_values = torch.atleast_2d(torch.tensor(control_values))
        n_data, n_control = control_values.shape
        # print("n_data, n_control",n_data, n_control)
        if control_data_labels is None:
            control_data_labels = [f'ctr{i}' for i in range(n_data)]
        if multi_state_objective_fn and n_data>1:
            n_sample = 1024*128
        else:
            n_sample = None
        result = msGP.predict(control_values,
                              n_sample = n_sample, 
                              multi_state_objective_fn = multi_state_objective_fn)
        
        n_data_pred, n_state, n_obj = result.mean.shape
        if objective_vals is not None:
            if objective_vals.ndim == 2:
                assert n_data == 1, f'shape of control_values indicate n_data is {n_data}. The shape of objective_vals must be (n_data, n_state, n_obj) but got {objective_vals.shape}'
                objective_vals = objective_vals[None,:,:]
            assert objective_vals.shape[1] == n_state
            assert objective_vals.shape[2] == n_obj
        assert n_data == n_data_pred
        assert n_state == len(state_names), f'number of state in msGP is {n_state} but lengh of state_names is {len(state_names)}'
        assert n_obj == len(objective_PVs), f'number of objective in msGP is {n_obj} but lengh of objective_PVs is {len(objective_PVs)}'

        nplot = n_obj
        if multi_state_objective_fn is not None:
            nplot = nplot + 1
        nrow = int(nplot/3)
        if 3*nrow < nplot:
            nrow += 1
        if nrow == 1:
            ncol = nplot
        else:
            ncol = 3
        fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol, 2.5*nrow))
        ax = ax.ravel()  # Flatten the axes array for easy indexing
        xaxis = np.arange(n_state)
        for i_obj in range(n_obj):
            for i_data in range(n_data):
                if objective_vals is not None:
                    ax[i_obj].plot(xaxis, objective_vals[i_data,:,i_obj],'k',label='True')
                label = control_data_labels[i_data] if n_data>1 else 'GP mean'
                ax[i_obj].plot(xaxis, result.mean[i_data,:,i_obj], label=label)
                ax[i_obj].fill_between(xaxis, 
                                       result.mean[i_data,:,i_obj] - CL * result.std[i_data,:,i_obj], 
                                       result.mean[i_data,:,i_obj] + CL * result.std[i_data,:,i_obj], 
                                       alpha=0.5)#, label=f'{CL}σ')
            ax[i_obj].set_title(objective_PVs[i_obj])
            ax[i_obj].legend()
            ax[i_obj].set_xticks(list(range(n_state)))
            ax[i_obj].set_xticklabels(state_names)

        # Plot objective
        # print("multi_state_objective_fn",multi_state_objective_fn is None)
        # print("n_data",n_data)
        if multi_state_objective_fn is not None and n_data>1:
            i_obj += 1
            obj_mean    = result.obj_samples.mean(dim=0).numpy()
            obj_std     = result.obj_samples.std (dim=0).numpy()
            xaxis = np.arange(n_data)
            if objective_vals is not None:
                ax[i_obj].scatter(xaxis, multi_state_objective_fn(objective_vals), c='k', label='True')
            #ax[i_obj].scatter(xaxis, multi_state_objective_fn(result.mean).numpy(), label='GP Mean')
            # ax[i_obj].errorbar(xaxis, obj_mean, yerr=CL * obj_std, alpha=0.5)
            ax[i_obj].plot(xaxis, obj_mean)
            ax[i_obj].fill_between(xaxis, obj_mean -CL*obj_std, obj_mean +CL*obj_std, alpha=0.5)#, label=f'{CL}σ')
            ax[i_obj].set_title('Objective')
            ax[i_obj].set_xticks(list(range(n_data)))
            ax[i_obj].set_xticklabels(control_data_labels)

        # if i_obj+1 < nrow*ncol:
        for empty_ax in ax[nplot:]:
            empty_ax.set_frame_on(False)
            empty_ax.set_xticks([])
            empty_ax.set_yticks([])
            
        plt.tight_layout()
    return fig,ax



# def plot(MTGT, CL=2, l_task_name=['48Ca19+','48Ca20+'], xlabel='QUAD:I_CSET', ylabel='BPM:XPOS'):
#     with torch.no_grad():
#         # Get posterior samples for both tasks
#         posterior = MTGT.model.posterior(grid_X)
#         pred_Y_samples = posterior.rsample(torch.Size([1024]))  # Draw 1024 samples from the posterior

#         # Separate the samples for task 0 and task 1
#         pred_Y_task0_samples = pred_Y_samples[:, :128, 0]  # 1024 x 128 (samples for task 0)
#         pred_Y_task1_samples = pred_Y_samples[:, 128:, 0]  # 1024 x 128 (samples for task 1)
        
#         # Compute the objective for each sample
#         obj_samples = objective(torch.stack([pred_Y_task0_samples, pred_Y_task1_samples], dim=-1))

#         # Compute mean and std of the objective across samples
#         obj_mean = obj_samples.mean(dim=0)  # Mean across samples
#         obj_std = obj_samples.std(dim=0)    # Standard deviation across samples

#         # Compute mean and std for task predictions
#         pred_Y_task0 = posterior.mean[:128].view(-1)
#         pred_Y_task1 = posterior.mean[128:].view(-1)
#         pred_Ystd_task0 = posterior.stddev[:128].view(-1)
#         pred_Ystd_task1 = posterior.stddev[128:].view(-1)

#         fig, ax = plt.subplots(1, 3, figsize=(18, 4))

#         # Plot task 0
#         ax[0].plot(grid_X_task0, grid_Y_task0, 'k', label='True')
#         ax[0].plot(grid_X_task0.view(-1), pred_Y_task0, label='GP mean')
#         ax[0].fill_between(grid_X_task0.view(-1), pred_Y_task0 - CL * pred_Ystd_task0, 
#                            pred_Y_task0 + CL * pred_Ystd_task0, alpha=0.5, label=f'{CL}σ')
#         ax[0].scatter(MTGT.l_train_X[0].view(-1), MTGT.l_train_Y[0].view(-1), color='r', label='Train data')
        
#         # Plot task 1
#         ax[1].plot(grid_X_task0, grid_Y_task1, 'k', label='True')
#         ax[1].plot(grid_X_task0.view(-1), pred_Y_task1, label='GP mean')
#         ax[1].fill_between(grid_X_task0.view(-1), pred_Y_task1 - CL * pred_Ystd_task1, 
#                            pred_Y_task1 + CL * pred_Ystd_task1, alpha=0.5, label=f'{CL}σ')
#         ax[1].scatter(MTGT.l_train_X[1].view(-1), MTGT.l_train_Y[1].view(-1), color='r', label='Train data')

#         # Plot objective
#         ax[2].plot(grid_X_task0, objective(torch.cat([grid_Y_task0, grid_Y_task1], dim=1)), 'k', label='True')
#         ax[2].plot(grid_X_task0, obj_mean, label='GP Mean')
#         ax[2].fill_between(grid_X_task0.view(-1), obj_mean - CL * obj_std, 
#                            obj_mean + CL * obj_std, alpha=0.5, label=f'{CL}σ')

#         # Efficiently find matching indices where task 0 and task 1 inputs are approximately equal
#         diff_matrix = torch.cdist(MTGT.l_train_X[0], MTGT.l_train_X[1])  # Compute pairwise distances
#         threshold = 1e-4  # Define a small threshold to consider the inputs as "matching"
#         matching_indices = torch.nonzero(diff_matrix < threshold, as_tuple=False)

#         # Gather matching X, Y0, Y1 based on matching indices
#         if matching_indices.size(0) > 0:  # Check if there are any matching pairs
#             x_match = MTGT.l_train_X[0][matching_indices[:, 0]]
#             y0_match = MTGT.l_train_Y[0][matching_indices[:, 0]]
#             y1_match = MTGT.l_train_Y[1][matching_indices[:, 1]]
#             y_obj_match = objective(torch.cat([y0_match, y1_match], dim=1))
            
#             # Scatter plot the matching data in the objective plot
#             ax[2].scatter(x_match.view(-1), y_obj_match, color='r', label='Train Data (Matching)')
        
#         # Add labels and legends
#         for a in ax:
#             a.legend()
        
#         ax[0].set_title(l_task_name[0])
#         ax[1].set_title(l_task_name[1])
#         ax[0].set_xlabel(xlabel)
#         ax[0].set_ylabel(ylabel)
#         ax[1].set_xlabel(xlabel)
#         ax[2].set_title('Objective')
#         ax[2].set_xlabel(xlabel)
#         ax[2].set_ylabel('objective value')

#         plt.tight_layout()
#         plt.show()
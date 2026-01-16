from typing import List, Union, Optional, Callable
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

import botorch
from botorch.models import MultiTaskGP as BotorchMultiTaskGP
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
from .construct_machineIO import Evaluator, MultiStateEvaluator, add_column_to_df
from .objective import MultiTaskIndexing, CompositeObjective


class MultiTaskBO:
    def __init__(self,
        machineIO,
        control_CSETs: List[str],
        control_RDs  : List[str],
        control_tols : Union[List[float], np.ndarray],
        control_min  : Union[List[float], np.ndarray],
        control_max  : Union[List[float], np.ndarray],
        monitor_PVs  : List[str],
        multi_task_objective_fn: CompositeObjective,
        task_names   : Optional[List[str]] = None,
        monitor_PVs_2_tasks: Optional[add_column_to_df] = None,
        control_couplings: Optional[dict] = None,
        local_optimization : Optional[bool] = False,
        local_bound_size   : Optional[Union[List[float], np.ndarray]] = None,
        use_ramping_model : Optional[bool] = False,
        MultiTaskIndexer: Optional[MultiTaskIndexing] = None,  # just to verify consistency of task_names of MultiTaskIndexer and MultiTaskBO
        ):
        """
        multi_task_objective_fn: should take tensor of shape (n_data, n_obj)
        """
        machineIO.clear_history()
        self.machineIO = machineIO
        self.control_CSETs = list(control_CSETs)
        self.control_RDs   = control_RDs
        self.control_tols  = control_tols
        self.control_min   = np.array(control_min)
        self.control_max   = np.array(control_max)
        self.n_control = len(control_CSETs)
        self.control_couplings = control_couplings
        self.monitor_PVs   = list(monitor_PVs)
        if task_names is None:
            self.task_names = self.monitor_PVs
        else:
            self.task_names = list(task_names)
        if monitor_PVs_2_tasks is None:
            monitor_PVs_2_tasks = add_column_to_df()
        self.monitor_PVs_2_tasks = monitor_PVs_2_tasks
        assert set(self.monitor_PVs_2_tasks.input_column_names) < set(self.control_CSETs + self.monitor_PVs)
        assert set(self.task_names) < set(self.monitor_PVs_2_tasks.output_column_names + self.control_CSETs + self.monitor_PVs)

        self.n_task  = len(self.task_names)
        self.local_optimization = local_optimization
        self.use_ramping_model = use_ramping_model

        if hasattr(multi_task_objective_fn,'task_names'):
            assert multi_task_objective_fn.task_names == self.task_names
        self.multi_task_objective_fn = multi_task_objective_fn
        self.mc_objective  = GenericMCObjective(multi_task_objective_fn)
        
        self.mtgp = None
        
        if local_optimization:
            if local_bound_size is None:
                self.local_bound_size = 0.1*(control_max - control_min)
            else: 
                self.local_bound_size = local_bound_size
        
        self.evaluator = Evaluator(
            machineIO     = machineIO,
            control_CSETs = self.control_CSETs,
            control_RDs   = self.control_RDs,
            control_tols  = self.control_tols,
            monitor_PVs   = self.monitor_PVs,
            control_couplings = self.control_couplings,
            column_adders = [self.monitor_PVs_2_tasks],
            )

        x0, df = self.machineIO.fetch_data(self.control_CSETs,time_span=0.2)
        # self.machineIO.clear_history()
        self._x0 = np.array(x0)
        assert self._x0.shape == (self.n_control,)
        
        self.timecost = {'query':[],'fit':[], 'fit_ramping':[]}

        if MultiTaskIndexer is not None:
            assert MultiTaskIndexer.task_names == self.task_names

        self.history = {'x':[],
                        'y':[],
                        'yvar':[],
                        'x_ramp':[],
                        'y_ramp':[]}
        self._x = self.history['x']
        self._y = self.history['y']
        self._yvar = self.history['yvar']
        self._x_ramp = self.history['x_ramp']
        self._y_ramp = self.history['y_ramp']
        self.ramping_model = None

    def update_ramping_model(self):
        t0 = time.monotonic()
        if self.ramping_model is None:
            self.ramping_model = MultiTaskGP(self.n_control, self.n_task)
        self.ramping_model.replace_data(x=self.x_ramp, y=self.y_ramp)
        self.ramping_model.fit()
        self.timecost['fit_ramping'].append(time.monotonic()-t0)

    def update_model(self):
        t0 = time.monotonic()
        if self.mtgp is None:
            self.mtgp = MultiTaskGP(self.n_control, self.n_task, prior_mean_model=self.ramping_model)
        self.mtgp.prior_mean_model = self.ramping_model
        self.mtgp.replace_data(x=self.x, y=self.y)
        self.mtgp.fit()
        self.timecost['fit'].append(time.monotonic()-t0)

    @property
    def control_bounds(self):
        return np.array(list(zip(self.control_min, self.control_max)))
    @property
    def control_bounds_botorch(self):
        return torch.tensor(np.array([self.control_min,self.control_max]))
    @property
    def x(self):
        return torch.stack(self._x, dim=0)
    @property
    def y(self):
        return torch.stack(self._y, dim=0)
    @property
    def yvar(self):
        return torch.stack(self._yvar, dim=0)
    @property
    def x_ramp(self):
        return torch.stack(self._x_ramp, dim=0)
    @property
    def y_ramp(self):
        return torch.stack(self._y_ramp, dim=0)

    def init(self,
             n_init,
             init_bound_size = None):

        if self.local_optimization:        
            lower_bound = np.maximum(self._x0 -0.5*self.local_bound_size, self.control_min)
            upper_bound = np.minimum(self._x0 +0.5*self.local_bound_size, self.control_max)
            bounds = np.vstack((lower_bound, upper_bound)).T  # Shape bounds as (n, 2)
        else:
            bounds = self.control_bounds

        init_x = proximal_ordered_init_sampler(
            n_init-1,
            bounds=bounds,
            x0=self._x0,
            ramping_rate=None,
            polarity_change_time=None, 
            method='sobol',
            seed=None)
        init_x = np.vstack((self._x0.reshape(1,-1),init_x))

        for isample,x in enumerate(init_x[:-1]):
            future = self.evaluator.submit(x)
            df, ramping_df = self.evaluator.get_result(future)
            if self.monitor_PVs_2_tasks is not None:
                df = self.monitor_PVs_2_tasks(df)
                ramping_df = self.monitor_PVs_2_tasks(ramping_df)
            # display(ramping_df) # debug
            self._x.append(torch.tensor(df[self.control_RDs].mean().values, dtype=torch.double))
            self._y.append(torch.tensor(df[self.task_names].mean().values, dtype=torch.double))
            self._yvar.append(torch.tensor(df[self.task_names].var().values, dtype=torch.double))
            self._x_ramp.append(torch.tensor(ramping_df[self.control_RDs].values, dtype=torch.double))
            self._y_ramp.append(torch.tensor(ramping_df[self.task_names].values, dtype=torch.double))

        self.future = self.evaluator.submit(init_x[-1])
        if self.use_ramping_model:          
            self.update_ramping_model()
        self.update_model()


    def get_acq_function(self,beta=4,X_pending=None):
        if beta < 0.01:
            X_pending = None
        else:
            if X_pending is not None:
                X_pending = torch.atleast_2d(X_pending)
                assert X_pending.shape[1] == self.n_control
        
            if X_pending is None:
                X_pending = self.mtgp.data_converter.lx2mtx(self.mtgp.lx)[:,:-1]
            else:
                # print("X_pending.shape",X_pending.shape)
                X_pending = self.mtgp.data_converter.x2mtx(X_pending)[:,:-1]
                # print("X_pending.shape",X_pending.shape)
            
        acq_function = qUpperConfidenceBound(
            model=self.mtgp.model,
            beta = beta,
            objective=self.mc_objective,
            X_pending = X_pending
        )
        return acq_function


    def query_candidate(self,bounds,acq_function=None,beta=4):
        t0 = time.monotonic()
        if acq_function is None:
            acq_function = qUpperConfidenceBound(
                model=self.mtgp.model,
                objective=self.mc_objective,
                beta=beta,
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

    def optimize_global(self, beta=4, X_pending=None):
        
        acq_function = self.get_acq_function(beta=beta, X_pending=None)        
        candidate = self.query_candidate(
            bounds=self.control_bounds_botorch, 
            acq_function=acq_function)

        df, ramping_df = self.evaluator.get_result(self.future)
        self._x.append(torch.tensor(df[self.control_RDs].mean().values, dtype=torch.double))
        self._y.append(torch.tensor(df[self.task_names].mean().values, dtype=torch.double))
        self._yvar.append(torch.tensor(df[self.task_names].var().values, dtype=torch.double))
        self._x_ramp.append(torch.tensor(ramping_df[self.control_RDs].values, dtype=torch.double))
        self._y_ramp.append(torch.tensor(ramping_df[self.task_names].values, dtype=torch.double))

        self.future = self.evaluator.submit(candidate.view(-1).detach().cpu().numpy())

        if self.use_ramping_model:          
            self.update_ramping_model()
        self.update_model()

    def finalize(self):
        df, ramping_df = self.evaluator.get_result(self.future)
        self._x.append(torch.tensor(df[self.control_RDs].mean().values, dtype=torch.double))
        self._y.append(torch.tensor(df[self.task_names].mean().values, dtype=torch.double))
        self._yvar.append(torch.tensor(df[self.task_names].var().values, dtype=torch.double))
        self._x_ramp.append(torch.tensor(ramping_df[self.control_RDs].values, dtype=torch.double))
        if self.use_ramping_model:          
            self.update_ramping_model()
        self.update_model()

    def plot_tasks(self):
        n_row = int(self.n_task/3)
        if 3*n_row < self.n_task:
            n_row += 1
        if n_row == 1:
            n_col = self.n_task
        else:
            n_col = 3
        y = torch.stack(self._y, dim=0)
        assert y.shape[1] == self.n_task
        fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 2.5*n_row))
        ax = ax.ravel()  # Flatten the axes array for easy indexing
        for i in range(self.n_task):
            ax[i].plot(y[:,i])
            ax[i].set_title(self.task_names[i])
        plt.tight_layout()
        plt.show()


    def plot_composite_objective(self, composite_objective_fn=None):
        if composite_objective_fn is None:
            composite_objective_fn = self.multi_task_objective_fn
        if hasattr(composite_objective_fn,'objfuncs') and hasattr(composite_objective_fn,'y_index'):
            objfuncs = composite_objective_fn.objfuncs
            y_index  = composite_objective_fn.y_index
        else:
            objfuncs = []
            y_index = []
        if hasattr(composite_objective_fn,'output_names'):
            objfuncs_names = composite_objective_fn.output_names
        else:
            objfuncs_names = [f'obj{i+1}' for i in range(len(objfuncs))]
        n_plot = len(objfuncs) + 1
        n_row = int(n_plot/3)
        if 3*n_row < n_plot:
            n_row += 1
        if n_row == 1:
            n_col = n_plot
        else:
            n_col = 3
        fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 2.5*n_row))
        ax = ax.ravel()  # Flatten the axes array for easy indexing
        y = torch.stack(self._y, dim=0)
        for i in range(len(objfuncs)):
            obj = objfuncs[i](y[:,y_index[i]])
            ax[i].plot(obj,label=objfuncs_names[i])
            ax[i].legend()
        obj = composite_objective_fn(y)
        ax[-1].plot(obj,label='composite objective')
        ax[-1].legend()
        plt.tight_layout()
        plt.show()


class MultiTaskGP_data_converter:
    """
    list of torch.Tensor x,y are used to append data for performance consideration. 
    Allocating memory for new added data is expensive.
    """
    def __init__(self, input_dim, n_task):
        self.input_dim = input_dim
        self.n_task = n_task

    def x2lx(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double)
        assert x.shape[1] == self.input_dim
        return [x]*self.n_task
    def y2ly(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.double)
        assert y.shape[1] == self.n_task
        return [y[:,i_task].unsqueeze(-1) for i_task in range(self.n_task)]
    def xy2lxy(self, x, y):
        assert x.shape[0] == y.shape[0]
        lx = self.x2lx(x)
        ly = self.y2ly(y)
        return lx, ly

    def lx2mtx(self, lx):
        lmtx = [torch.cat([x, i * torch.ones(x.shape[0], 1)], dim=1) for i, x in enumerate(lx) if x is not None]
        mtx = torch.cat(lmtx, dim=0)
        return mtx
    def ly2mty(self, ly):
        lmty = [y for y in ly if y is not None]
        mty = torch.cat(lmty, dim=0)
        return mty
    def lxy2mtxy(self, lx, ly):
        assert len(lx) == len(ly)
        mtx = self.lx2mtx(lx)
        mty = self.ly2mty(ly)
        return mtx, mty

    def x2mtx(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double)
        x = x.contiguous()
        assert x.shape[1] == self.input_dim
        mtx_x = x.repeat(self.n_task, 1)  # Shape: (n_samples * n_tasks, n_features)
        task_indices = torch.arange(self.n_task).repeat_interleave(x.shape[0])
        mtx = torch.cat([mtx_x, task_indices.unsqueeze(1)], dim=1)  # Add task column
        return mtx
    def y2mty(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.double)
        y = y.contiguous()
        assert y.shape[1] == self.n_task
        mty = y.T.reshape(-1, 1)  # Equivalent to stacking columns vertically
        return mty
    def xy2mtxy(self, x, y):
        assert x.shape[0] == y.shape[0]
        mtx = self.x2mtx(x)
        mty = self.y2mty(y)
        return mtx, mty

    def mtx2lx(self, mtx):
        assert mtx.shape[1] == self.input_dim + 1
        col_task = mtx[:,-1]
        lx = []
        for i_task in range(self.n_task):
            mask = col_task==i_task
            lx.append(mtx[mask,:-1])
        return lx
    def mty2ly(self, mty, col_task):
        assert mty.shape[1] == 1
        ly = []
        for i_task in range(self.n_task):
            mask = col_task==i_task
            ly.append(mty[mask,0].unsqueeze(-1))
        return ly
    def mtxy2lxy(self, mtx, mty):
        assert mtx.shape[0] == mty.shape[0]
        col_task = mtx[:,-1]
        lx = self.mtx2lx(mtx)
        ly = self.mty2ly(mty, col_task)
        return lx, ly

    def ly2y(self, ly):
        n_data = ly[0].shape[0]
        assert all([y.shape[0] == n_data for y in ly])
        y = torch.cat([y for y in ly], dim=1)
        return y

    def append_task_column(self, x, i_task):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double)
        assert i_task < self.n_task, "Invalid task index."
        assert x.shape[1] == self.input_dim
        task_col = torch.ones(x.shape[0], 1)*i_task
        # Concatenate the tensor of ones to the last dimension of x
        return torch.cat([x, task_col], dim=-1)



class MultiTaskGP:
    def __init__(self,input_dim, n_task, prior_mean_model=None, GP_kwargs=None):
        """
        list of torch.Tensor x,y are used to append data for performance consideration. 
        Allocating memory for new added data is expensive.

        Args:
            
        """
        self.data_converter = MultiTaskGP_data_converter(input_dim, n_task)
        self.input_dim = self.data_converter.input_dim
        self.n_task = self.data_converter.n_task
        self.prior_mean_model = prior_mean_model
        if GP_kwargs is None:
            GP_kwargs = {}
        else:
            GP_kwargs.pop('task_feature', None)
            GP_kwargs.pop('input_transform', None)
            GP_kwargs.pop('outcome_transform', None)
        self.GP_kwargs = GP_kwargs
        

    def __call__(self, mtx):
        with torch.no_grad():
            # mtx = self.data_converter.x2mtx(x)
            xn = self.normalize_mtx(mtx)
            posterior = self.model.posterior(xn)
            mean = self.unstandardize_mty(posterior.mean)
            if self.prior_mean_model is not None:
                mty_prior = self.prior_mean_model(mtx)
                mean = mean + mty_prior
            return mean
        
    def fit_normalize(self, lx):
        with torch.no_grad():
            self.xmin = torch.min([x.min(dim=0).values for x in lx], dim=0).values  # Extract values
            self.xmaxmin = torch.max([x.max(dim=0).values for x in lx], dim=0).values - self.xmin  # Extract values and compute range
    def normalize(self, lx):
        with torch.no_grad():
            return [(x - self.xmin) / self.xmaxmin for x in lx]
    def unnormalize(self, lxn):
        with torch.no_grad():
            return [xn * self.xmaxmin + self.xmin for xn in lxn]
    def fit_standardize(self, ly):
        with torch.no_grad():
            self.ymean = [y.mean() for y in ly]
            self.ystd = [y.std() for y in ly]
    def standardize(self, ly): 
        with torch.no_grad():
            return torch.tensor([(y - ymean) / ystd for y, ymean, ystd in zip(ly,  self.ymean, self.ystd)])
    def unstandardize(self, lys):
        with torch.no_grad():
            return torch.tensor([ys * ystd + ymean for ys, ymean, ystd in zip(lys, self.ymean, self.ystd)])

    def fit_normalize_mtx(self, mtx):
        with torch.no_grad():
            self.xmin = torch.min(mtx[:,:-1], dim=0).values
            self.xmaxmin = torch.max(mtx[:,:-1], dim=0).values - self.xmin
    def normalize_mtx(self, mtx):
        with torch.no_grad():
            xn = torch.empty((mtx.shape[0],self.input_dim+1),dtype=torch.double)
            xn[:,:-1] = (mtx[:,:-1] - self.xmin) / self.xmaxmin
            xn[:,-1] = mtx[:,-1]
            return xn
    def unnormalize_mtx(self, mtxn):
        with torch.no_grad():
            x = torch.empty((mtxn.shape[0],self.input_dim),dtype=torch.double)
            x[:,:-1] = mtxn[:,:-1] * self.xmaxmin + self.xmin
            x[:,-1] = mtxn[:,-1]
            return x
    def fit_standardize_mty(self, mty, task_col):
        with torch.no_grad():
            self.ymean = torch.zeros(self.n_task, device=mty.device)
            self.ystd = torch.ones(self.n_task, device=mty.device)  # Default std=1
            for i_task in range(self.n_task):
                mask = task_col == i_task
                if mask.any():
                    task_data = mty[mask, 0]
                    self.ymean[i_task] = task_data.mean()
                    std = task_data.std()
                    self.ystd[i_task] = std if std > 1e-14 else 1.0
    def standardize_mty(self, mty, task_col):
        with torch.no_grad():
            ys = mty.clone()
            for i_task in range(self.n_task):
                mask = task_col == i_task
                if mask.any():
                    ys[mask, 0] = (mty[mask, 0] - self.ymean[i_task]) / self.ystd[i_task]
            return ys
    def unstandardize_mty(self, mty, task_col):
        with torch.no_grad():
            y = mty.clone()
            for i_task in range(self.n_task):
                mask = task_col == i_task
                if mask.any():
                    y[mask, 0] = mty[mask, 0] * self.ystd[i_task] + self.ymean[i_task]
            return y

    def replace_data(self, x=None, y=None, lx=None, ly=None):
        if x is not None and y is not None:
            self.lx, self.ly = self.data_converter.xy2lxy(x, y)
        elif lx is not None and ly is not None:
            assert len(lx) == len(ly) == self.n_task, "Invalid data length."
            self.lx = []
            self.ly = []
            for i_task in range(self.n_task):
                x = lx[i_task]
                y = ly[i_task]
                if not isinstance(x, torch.Tensor) and x is not None:
                    x = torch.tensor(x, dtype=torch.double)
                if not isinstance(y, torch.Tensor) and y is not None:
                    y = torch.tensor(y, dtype=torch.double)
                self.lx.append(x)
                self.ly.append(y)
        else:
            raise ValueError("Either x and y or lx and ly must be provided.")


    def append_new_data_each_task(self, i_task, candidate_x, evaluated_y):
        """
        Adds new data for a specific task, updates the internal storage, and re-combines the data.
        
        Args:
            i_task (int): Index of the task to which new data belongs.
            candidate_x (torch.Tensor): New input data for the task. shape of n_data, input_dim
            evaluated_y (torch.Tensor): New output data for the task. shape of n_data, 1
        """
        assert i_task < self.n_task, "Invalid task index."
        assert candidate_x.shape[0] == evaluated_y.shape[0], "Invalid data length."
        assert candidate_x.shape[1] == self.input_dim, "Invalid input dimension."
        assert evaluated_y.shape[1] == 1, "Invalid output dimension."
        self.lx[i_task] = torch.cat([self.lx[i_task], candidate_x], dim=0)
        self.ly[i_task] = torch.cat([self.ly[i_task], evaluated_y], dim=0)


    def append_new_data(self, candidate_x, evaluated_y):
        """
        Adds new data for a whole task, updates the internal storage, and re-combines the data.
        Args:
            i_task (int): Index of the task to which new data belongs.
            candidate_x (torch.Tensor): New input data for the task. shape of n_data, input_dim
            evaluated_y (torch.Tensor): New output data for the task. shape of n_data, n_task
        """
        assert candidate_x.shape[0] == evaluated_y.shape[0], "Invalid data length."
        assert candidate_x.shape[1] == self.input_dim, "Invalid input dimension."
        assert evaluated_y.shape[1] == self.n_task, "Invalid output dimension."
        for i_task in range(self.n_task):
            self.lx[i_task] = torch.cat([self.lx[i_task], candidate_x], dim=0)
            self.ly[i_task] = torch.cat([self.ly[i_task], evaluated_y[:,i_task].unsqueeze(-1)], dim=0)


    def fit(self,lr=0.2,epochs=200):
        mtx, mty = self.data_converter.lxy2mtxy(self.lx, self.ly)
        if self.prior_mean_model is not None:
            mty_prior = self.prior_mean_model(mtx)
            mty = mty - mty_prior

        self.fit_normalize_mtx(mtx)
        self.fit_standardize_mty(mty, mtx[:,-1])
        xn = self.normalize_mtx(mtx)
        ys = self.standardize_mty(mty, mtx[:,-1])

        model = BotorchMultiTaskGP(xn,  ys, 
            task_feature=-1,  # The last column indicates the task index
            **self.GP_kwargs,
            # mean_module=self.prior_mean_model,
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
        mtx = self.data_converter.x2mtx(x)
        xn = self.normalize_mtx(mtx)
        posterior = self.model.posterior(xn)

        #posterior.mean is shape of (len(x)*n_task,)
        mtys = posterior.mean
        mtys_std = posterior.stddev
        mty  = self.unstandardize_mty(mtys)
        mty_std = mtys_std * self.ystd.repeat_interleave(len(x))

        if self.prior_mean_model is not None:
            mty_prior = self.prior_mean_model(mtx)
            mty = mty + mty_prior

        ly = self.data_converter.mty2ly(mty, mtx[:,-1])
        y = self.data_converter.ly2y(ly)
        ly_std = self.data_converter.mty2ly(mty_std, mtx[:,-1])
        y_std = self.data_converter.ly2y(ly_std)
        
        result = dictClass()
        result.mean = y
        result.std  = y_std
        
        if n_sample:
            pred_Y_samples = posterior.rsample(torch.Size([n_sample])) #shape of (n_sample,len(x)*n_task)
            for isample in range(n_sample):
                result.samples[isample,:] = self.unstandardize_mty(pred_Y_samples[isample,:], mtx[:,-1])
        if multi_task_objective_fn:  
            # mc_objective = GenericMCObjective(multi_task_objective_fn)
            result.obj_mean = multi_task_objective_fn(result.mean)
            if n_sample:
                result.obj_samples = multi_task_objective_fn(result.samples)  # shape of (n_sample, len(x))
        return result

# class mtGP:
#     def __init__(self, n_task, input_dim, train_X=None, train_Y=None, prior_mean_model=None, GP_kwargs=None, multi_task_objective_fn=None):

#         self.n_task = n_task
#         self.input_dim = input_dim        
#         if GP_kwargs is None:
#             GP_kwargs = {}
#         else:
#             GP_kwargs.pop('task_feature', None)
#             GP_kwargs.pop('input_transform', None)
#             GP_kwargs.pop('outcome_transform', None)
#         self.GP_kwargs = GP_kwargs       

#         if train_X is not None:
#             assert train_X.shape[1] == self.input_dim + 1
#             self.train_X = train_X
#         else:
#             self.train_X = torch.empty(0, self.input_dim + 1)  # Initialize to empty tensor
    
#         if train_Y is not None:
#             assert train_Y.shape[1] == 1
#             self.train_Y = train_Y
#         else:
#             self.train_Y = torch.empty(0, 1)  # Initialize to empty tensor
#         self.prior_mean_model = prior_mean_model

#     def __call__(self, x):
#         xn = self.normalize(x)
#         posterior = self.model.posterior(xn)
#         mean = self.unstandardize(posterior.mean)
#         if self.prior_mean_model is not None:
#             prior_mean = self.prior_mean_model(x)
#             mean = mean + prior_mean
#         return mean

#     def fit_normalize(self, x):
#         self.xmin = x[:,:-1].min(dim=0).values  # Extract values
#         self.xmaxmin = x[:,:-1].max(dim=0).values - self.xmin  # Extract values and compute range
#     def normalize(self, x):
#         xn = (x[:,:-1] - self.xmin) / self.xmaxmin  # Fix slicing
#         return torch.cat([xn, x[:,-1:]], dim=-1)  # Preserve task column
#     def unnormalize(self, xn):
#         x = xn[:,:-1] * self.xmaxmin + self.xmin
#         return torch.cat([x, xn[:,-1:]], dim=-1)  # Preserve task column
#     def fit_standardize(self, y):
#         self.ymean = y.mean()
#         self.ystd = y.std()
#     def standardize(self, y):
#         return (y - self.ymean) / self.ystd
#     def unstandardize(self, ys):
#         return ys * self.ystd + self.ymean
        

#     def append_task_column(self, x, i_task):
#         assert i_task < self.n_task, "Invalid task index."
#         assert x.shape[1] == self.input_dim
#         task_col = torch.ones(x.shape[0], 1)*i_task
#         # Concatenate the tensor of ones to the last dimension of x
#         return torch.cat([x, task_col], dim=-1)
    
#     def append_new_train_data(self, x, y, i_task=None):
#         """
#         Adds new data for a specific task
        
#         Args:
#             i_task (int): Index of the task to which new data belongs.
#             x (torch.Tensor): New input data for the task.
#             y (torch.Tensor): New output data for the task.
#         """
#         # Concatenate the tensor of ones to the last dimension of x
#         assert y.shape[0] ==x.shape[0]
#         if i_task is None:
#             assert x.shape[1] == self.input_dim
#         else:
#             assert x.shape[1] == self.input_dim + 1
#             x = self.append_task_column(x, i_task)
#         self.train_X = torch.cat([self.train_X, x], dim=0)
#         self.train_Y = torch.cat([self.train_Y, y], dim=0)  
        
        
#     def fit(self,lr=0.2,epochs=200):
#         if self.prior_mean_model is not None:
#             y_prior = self.prior_mean_model(self.train_X)
#             y = self.train_Y - y_prior
#         else:
#             y = self.train_Y
#         self.fit_normalize(self.train_X)
#         self.fit_standardize(y)
#         xn = self.normalize(self.train_X)
#         ys = self.standardize(y)
        
#         model = MultiTaskGP(
#             xn,ys,task_feature=-1,  # The last column indicates the task index
#             **self.GP_kwargs,
#             )
#         mll = ExactMarginalLogLikelihood(model.likelihood, model)
#         model.train()
#         # fit_gpytorch_mll(mll, options={"ftol": 1e-6, "maxiter": 200})
#         opt = torch.optim.Adam(mll.parameters(), lr=lr)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
#                                                         max_lr=lr,
#                                                         div_factor=2.5,
#                                                         pct_start=0.1, 
#                                                         final_div_factor=5,
#                                                         epochs=epochs, steps_per_epoch=1)
#         if not hasattr(self,"train_history"):
#             self.train_history = []
#         self.train_history.append([])
        
#         for _ in range(epochs):  # Number of iterations
#             opt.zero_grad()
#             model_output = model(*model.train_inputs)
#             log_likelihood = mll(
#                 model_output,
#                 model.train_targets,
#                 *(model.transform_inputs(X=t_in) for t_in in model.train_inputs),
#             )
#             loss = -log_likelihood.mean()
#             loss.backward()
#             opt.step()
#             scheduler.step()
#             self.train_history[-1].append(loss.item())
#         model.eval()
#         self.model = model

                
#     def posterior(self, x, i_task=None, n_sample=None):
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x)
#         if i_task is None:
#             assert x.shape[1] == self.input_dim + 1
#         else:
#             assert x.shape[1] == self.input_dim
#             x = self.append_task_column(x, i_task)
#         xn = self.normalize(x)
#         posterior = self.model.posterior(xn)
#         mean = self.unstandardize(posterior.mean)
#         if self.prior_mean_model is not None:
#             prior_mean = self.prior_mean_model(x)
#             mean = mean + prior_mean
#         std  = posterior.std * self.ystd
#         if n_sample is not None:
#             ys_samples = posterior.rsample(sample_shape=torch.Size([n_sample]))
#             y_samples = self.unstandardize(ys_samples)
#             if self.prior_mean_model is not None:
#                 y_samples = y_samples + prior_mean.unsqueeze(0)  # y_samples is (n_sample, n_data)?
#         return {'mean':mean,'std':std,'samples':y_samples}

        # result = dictClass() 
        # result.mean = self.unstandardize(posterior.mean)
        # result.std = self.unstandardize(posterior.std)
        # if n_sample is not None:
        #     ys_samples = posterior.rsample(sample_shape=torch.Size([n_sample]))
        #     result.samples = self.unstandardize(ys_samples)
        # return result



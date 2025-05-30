from typing import List, Union, Optional, Callable
import numpy as np
import logging
import torch

class MultiTaskIndexing:
    def __init__(self,
                 task_names: List[str],
                 ):
        self.task_names = list(task_names)

    def __call__(self, task_names: str):
        return self.get_task_index(task_names)
        
    def get_task_index(self, task_names):
        return [self.task_names.index(pv) for pv in task_names]
        

class CompositeObjective:
    def __init__(self, objfuncs, y_index, weights=None, input_names=None, output_names=None):
        assert len(y_index) == len(objfuncs), "Mismatch between number of tasks and index groups."
        self.objfuncs = objfuncs
        self.y_index = y_index
        if input_names is not None:
            assert len(input_names) == len(objfuncs), "Mismatch between number of tasks and index groups."
            for y_i, name_i in zip(y_index, input_names):
                assert len(name_i) == len(y_i), "Mismatch between number of tasks and index groups."
            self.input_names = list(input_names)
        if output_names is not None:
            assert len(objfuncs) == len(output_names), "Mismatch between number of tasks and index groups."
            self.output_names = list(output_names)
        
        self.n_obj = len(objfuncs)  
        assert self.n_obj == len(y_index), "Mismatch between number of tasks and index groups."

        if weights is None:
            self.weights = torch.ones(len(objfuncs)) / self.n_obj
        else:
            assert self.n_obj == len(weights), "Weights should match number of tasks."
            if not isinstance(weights, torch.Tensor):  
                with torch.no_grad():
                    self.weights = torch.tensor(weights, dtype=torch.float32)
                    self.weights = self.weights / self.weights.sum()
            else:
                self.weights = weights / weights.sum()

    def __call__(self, y, X=None):
        # print("y.shape",y.shape)
        # print("y",y)
        # objs = torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)])
        # print("objs.shape",objs.shape)
        # print("objs",objs)
        # objtot = torch.sum(objs,dim=0)  
        # print('objtot.shape',objtot.shape)    
        # print('objtot',objtot)  
        # return objtot
        return torch.sum(
            torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)]),
            dim=0
        )



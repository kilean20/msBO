from typing import List, Union, Optional, Callable, Tuple
import numpy as np
import logging
import torch
from torch import Tensor
import torch.nn.functional as F



def task_id(s: int, j: int, J: int) -> int:
    return s * J + j

def split_task(t: int, J: int) -> Tuple[int, int]:
    s = t // J
    j = t % J
    return s, j


def softmin(x, tau: float = 1e-3, dim: int = -1):
    # = -softmax(x/τ)^T x when τ→0 → min
    w = torch.softmax(-x / tau, dim=dim)
    return (w * x).sum(dim=dim)

# Element-wise ELU-based threshold around 0.95
def elu_step(y, center=0.95, width=0.05):
    scaled = (y - center) / width
    return F.elu(scaled) + 1  # maps: y << center → ~1, y >> center → ~0


class BPMvar_minimization:
    def __init__(self, S: int, J: int, w_beam_center = 0.2, w_beam_loss=1.0):
        '''
        S : number of states
        J : number of tasks for each state. 
            Important: All task must be BPM pos except last task. Last task is BPM MAG based contraint. 
        '''
        self.S, self.J = S, J
        self.w_beam_center = w_beam_center
        self.w_beam_loss = w_beam_loss

    def _index(self, samples: Tensor, s: int, j: int) -> Tensor:
        # samples: [..., q, T] ; returns [..., q]
        t = task_id(s, j, self.J)
        return samples[..., t]

    def __call__(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # stack across states → shape [..., q, S]
        ys = [torch.stack([self._index(samples, s, t) for s in range(self.S)], dim=-1) for t in range(self.J)]
        losses = [
                                 torch.var (ys[t],    dim=-1) +   # (..., q)
            self.w_beam_center * torch.mean(ys[t]**2, dim=-1)     # (..., q)
            for t in range(self.J - 1)
        ]
        if len(losses) > 0:
            # stack -> (..., q, J-1), then sum over the last dim -> (..., q)
            loss_sum = torch.stack(losses, dim=-1).sum(dim=-1)
        else:
            # no variance tasks; create a zero tensor matching the shape we’ll add it to
            loss_sum = torch.zeros_like(ys[-1].min(dim=-1).values)
        is_no_beamloss = elu_step(-softmin(ys[-1], tau=1e-2, dim=-1), center=0.95, width=0.05)  # cetner:0.95 and width:0.05 means 5% and 10% beam loss corresponds to 0.5, and 1 contraint penality respectivly

        return 1.0 - loss_sum + is_no_beamloss


# class MultiTaskIndexing:
#     def __init__(self,
#                  task_names: List[str],
#                  ):
#         self.task_names = list(task_names)

#     def __call__(self, task_names: str):
#         return self.get_task_index(task_names)
        
#     def get_task_index(self, task_names):
#         return [self.task_names.index(pv) for pv in task_names]
        

# class CompositeObjective:
#     def __init__(self, objfuncs, y_index, weights=None, input_names=None, output_names=None):
#         assert len(y_index) == len(objfuncs), "Mismatch between number of tasks and index groups."
#         self.objfuncs = objfuncs
#         self.y_index = y_index
#         if input_names is not None:
#             assert len(input_names) == len(objfuncs), "Mismatch between number of tasks and index groups."
#             for y_i, name_i in zip(y_index, input_names):
#                 assert len(name_i) == len(y_i), "Mismatch between number of tasks and index groups."
#             self.input_names = list(input_names)
#         if output_names is not None:
#             assert len(objfuncs) == len(output_names), "Mismatch between number of tasks and index groups."
#             self.output_names = list(output_names)
        
#         self.n_obj = len(objfuncs)  
#         assert self.n_obj == len(y_index), "Mismatch between number of tasks and index groups."

#         if weights is None:
#             self.weights = torch.ones(len(objfuncs)) / self.n_obj
#         else:
#             assert self.n_obj == len(weights), "Weights should match number of tasks."
#             if not isinstance(weights, torch.Tensor):  
#                 with torch.no_grad():
#                     self.weights = torch.tensor(weights, dtype=torch.float32)
#                     self.weights = self.weights / self.weights.sum()
#             else:
#                 self.weights = weights / weights.sum()

#     def __call__(self, y, X=None):
#         # print("y.shape",y.shape)
#         # print("y",y)
#         # objs = torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)])
#         # print("objs.shape",objs.shape)
#         # print("objs",objs)
#         # objtot = torch.sum(objs,dim=0)  
#         # print('objtot.shape',objtot.shape)    
#         # print('objtot',objtot)  
#         # return objtot
#         return torch.sum(
#             torch.stack([self.weights[i] * self.objfuncs[i](y[..., self.y_index[i]]) for i in range(self.n_obj)]),
#             dim=0
#         )



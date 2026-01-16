from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor


def task_id(s: int, j: int, J: int) -> int:
    return s * J + j

def split_task(t: int, J: int) -> Tuple[int, int]:
    s = t // J
    j = t % J
    return s, j



def _expand_tasks_all_states(
    X_ctrl: Tensor, T: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Expand control-only X (b x m x d) to all tasks 0..T-1 as task-feature col.
       Returns (b x (m*T) x (d+1)).
    """
    b, m, d = X_ctrl.shape
    t_ids = torch.arange(T, device=device, dtype=dtype).view(1, 1, T, 1)      # 1x1xT×1
    Xrep  = X_ctrl.unsqueeze(2).repeat(1, 1, T, 1)                            # b×m×T×d
    Xt    = torch.cat([Xrep, t_ids.repeat(b, m, 1, 1)], dim=-1)               # b×m×T×(d+1)
    return Xt.view(b, m*T, d+1)                                               # b×(mT)×(d+1)


def _expand_tasks_fixed_state(
    X_ctrl: Tensor, J: int, s_idx: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Expand control-only X (b x q x d) to the J outputs of ONE fixed state s_idx.
       Task ids are s_idx*J .. s_idx*J+J-1. Returns (b x (q*J) x (d+1)).
    """
    b, q, d = X_ctrl.shape
    base = s_idx * J
    t_ids = torch.arange(base, base + J, device=device, dtype=dtype).view(1, 1, J, 1)
    Xrep  = X_ctrl.unsqueeze(2).repeat(1, 1, J, 1)                             # b×q×J×d
    Xt    = torch.cat([Xrep, t_ids.repeat(b, q, 1, 1)], dim=-1)                # b×q×J×(d+1)
    return Xt.view(b, q*J, d+1)                                                # b×(qJ)×(d+1)


class MultiStateDataset:
    """
    Raw storage (per-oracle call):
      - self._x: list[Tensor], each (x_dim,)
      - self._y: list[Tensor], each (J,)
      - self._s: list[int],    each in [0, S)
      - self._yvar: list[Tensor], each (J,) only if provided for ALL rows

    BoTorch MultiTaskGP training views (computed lazily & cached):
      - X: (N, x_dim+1), last column is task index t = s*J + j  (float)
      - Y: (N, 1)
      - Yvar: (N, 1) if provided for all oracles, else empty (0,1)
      where N = len(self._x) * J
    """
    def __init__(self, x_dim: int, S: int, J: int, dtype=None, device=None):
        self.x_dim = x_dim
        self.S, self.J = S, J
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._device = torch.device(device) if device is not None else torch.device("cpu")

        # Raw per-oracle lists
        self._x: List[Tensor] = []
        self._y: List[Tensor] = []
        self._s: List[int]    = []
        self._yvar: List[Tensor] = []

        # Whether every appended row has yvar (once False, stays False)
        self._expect_yvar_for_all: Optional[bool] = None

        # Cached training views
        self._dirty = True
        self._X_cache: Optional[Tensor] = None
        self._Y_cache: Optional[Tensor] = None
        self._Yv_cache: Optional[Tensor] = None

    @staticmethod
    def _ensure_2d(x: Tensor, last_dim: int, name: str) -> Tensor:
        if x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() != 2:
            raise ValueError(f"{name}: expected 1D or 2D tensor, got {tuple(x.shape)}")
        if x.size(-1) != last_dim:
            raise ValueError(f"{name}: expected last dim {last_dim}, got {x.size(-1)} for shape {tuple(x.shape)}")
        return x

    @staticmethod
    def _ensure_1d_int(s: Union[int, Tensor], n: int, name: str) -> Tensor:
        if isinstance(s, int):
            s = torch.full((n,), s, dtype=torch.long)
        elif isinstance(s, Tensor):
            if s.dim() == 0:
                s = s.view(1).to(dtype=torch.long)
            elif s.dim() == 1:
                s = s.to(dtype=torch.long)
            else:
                raise ValueError(f"{name}: expected scalar or 1D tensor, got {tuple(s.shape)}")
            if s.numel() not in (1, n):
                raise ValueError(f"{name}: length {s.numel()} must be 1 or match batch size {n}")
            if s.numel() == 1:
                s = s.expand(n)
        else:
            raise TypeError(f"{name}: expected int or Tensor, got {type(s)}")
        return s

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._X_cache = None
        self._Y_cache = None
        self._Yv_cache = None

    # ---------- raw append ----------

    def concat_data(
        self,
        x: Tensor,                # (x_dim,) or (n, x_dim)
        s: Union[int, Tensor],    # scalar or (n,)
        y: Tensor,                # (J,) or (n, J)
        yvar: Optional[Tensor] = None  # (J,) or (n, J)
    ) -> None:
        """
        Append oracle results to raw lists. Accepts single or batched inputs,
        but stores *per-oracle* entries: (x_i: (x_dim,), s_i: int, y_i: (J,))
        """
        x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        y = torch.as_tensor(y, device=self._device, dtype=self._dtype)

        x2 = self._ensure_2d(x, self.x_dim, "x")   # (n, x_dim)
        y2 = self._ensure_2d(y, self.J, "y")       # (n, J)
        if x2.size(0) != y2.size(0):
            raise ValueError(f"batch mismatch: x has {x2.size(0)} rows, y has {y2.size(0)}")

        n = x2.size(0)
        s_vec = self._ensure_1d_int(s, n, "s")     # (n,)

        if (s_vec < 0).any() or (s_vec >= self.S).any():
            bad = s_vec[(s_vec < 0) | (s_vec >= self.S)]
            raise ValueError(f"s out of range [0,{self.S}): {bad.tolist()}")

        if yvar is not None:
            yvar = torch.as_tensor(yvar, device=self._device, dtype=self._dtype)
            yvar2 = self._ensure_2d(yvar, self.J, "yvar")
            if yvar2.size(0) != n:
                raise ValueError(f"batch mismatch: yvar has {yvar2.size(0)} rows, expected {n}")
            provide_yvar = True
        else:
            yvar2 = None
            provide_yvar = False

        # Enforce "all-or-nothing" yvar policy
        if self._expect_yvar_for_all is None:
            self._expect_yvar_for_all = provide_yvar
        else:
            if self._expect_yvar_for_all != provide_yvar:
                raise RuntimeError(
                    "Inconsistent yvar: Some rows provide noise and others do not. "
                    "Provide yvar for ALL rows or NONE."
                )

        for i in range(n):
            self._x.append(x2[i].clone())                 # (x_dim,)
            self._y.append(y2[i].clone())                 # (J,)
            self._s.append(int(s_vec[i].item()))          # int
            if provide_yvar:
                self._yvar.append(yvar2[i].clone())       # (J,)

        self._mark_dirty()

    # ---------- cached training views ----------

    @property
    def X(self) -> Tensor:
        """(N, x_dim+1), last column is task index (float)."""
        if not self._dirty and self._X_cache is not None:
            return self._X_cache

        m = len(self._x)
        if m == 0:
            out = torch.empty(0, self.x_dim + 1, dtype=self._dtype, device=self._device)
            self._X_cache = out
            return out

        Xc = torch.stack(self._x, dim=0)                                  # (m, x_dim)
        sc = torch.tensor(self._s, device=self._device, dtype=torch.long)  # (m,)

        # Repeat each row J times
        X_rep = Xc.repeat_interleave(self.J, dim=0)                        # (m*J, x_dim)

        # Task ids 0..J-1 for each oracle, then add s*J
        j = torch.arange(self.J, device=self._device, dtype=torch.long)    # (J,)
        t = (sc.unsqueeze(1) * self.J + j.unsqueeze(0)).reshape(-1, 1)     # (m*J, 1)
        Xt = torch.cat([X_rep, t.to(self._dtype)], dim=-1)                 # float last col

        self._X_cache = Xt
        return Xt

    @property
    def Y(self) -> Tensor:
        """(N, 1) scalar targets aligned with X row-wise."""
        if not self._dirty and self._Y_cache is not None:
            return self._Y_cache

        m = len(self._y)
        if m == 0:
            out = torch.empty(0, 1, dtype=self._dtype, device=self._device)
            self._Y_cache = out
            return out

        Yc = torch.stack(self._y, dim=0).reshape(m * self.J, 1)            # (m*J, 1)
        self._Y_cache = Yc
        return Yc

    @property
    def Yvar(self) -> Tensor:
        """
        (N, 1) if yvar provided for every oracle; otherwise empty (0,1).
        """
        if not self._dirty and self._Yv_cache is not None:
            return self._Yv_cache

        m = len(self._x)
        if m == 0 or not self._expect_yvar_for_all:
            out = torch.empty(0, 1, dtype=self._dtype, device=self._device)
            self._Yv_cache = out
            return out

        if len(self._yvar) != m:
            # Should never happen with strict policy above; guard anyway.
            out = torch.empty(0, 1, dtype=self._dtype, device=self._device)
            self._Yv_cache = out
            return out

        Yvc = torch.stack(self._yvar, dim=0).reshape(m * self.J, 1)        # (m*J, 1)
        self._Yv_cache = Yvc
        return Yvc

    # ---------- housekeeping ----------

    def clear(self) -> None:
        self._x.clear()
        self._y.clear()
        self._s.clear()
        self._yvar.clear()
        self._expect_yvar_for_all = None
        self._dirty = False
        self._X_cache = None
        self._Y_cache = None
        self._Yv_cache = None

    def __len__(self) -> int:
        """Number of training rows N = (#oracles) * J."""
        return len(self._x) * self.J

    def indices_by_state(self, s: int) -> List[int]:
        """
        Return the list of raw-oracle indices whose state equals `s`.
        """
        if not (0 <= s < self.S):
            raise ValueError(f"state s={s} out of range [0,{self.S})")
        return [k for k, sk in enumerate(self._s) if sk == s]

    def get_state_data(self, s: int, stack: bool = True):
        """
        Extract raw oracle data for a single state `s`.

        Returns:
            If stack=True:
                x_s    : Tensor of shape (m, x_dim)
                y_s    : Tensor of shape (m, J)
                yvar_s : Tensor of shape (m, J) if available for all; else None
                idx    : List[int] indices into the raw lists
            If stack=False:
                x_list : List[Tensor], each (x_dim,)
                y_list : List[Tensor], each (J,)
                yv_list: List[Tensor] or empty list if unavailable
                idx    : List[int]
        """
        idx = self.indices_by_state(s)
        if len(idx) == 0:
            if stack:
                x_empty = torch.empty(0, self.x_dim, dtype=self._dtype, device=self._device)
                y_empty = torch.empty(0, self.J,     dtype=self._dtype, device=self._device)
                return x_empty, y_empty, None, []
            else:
                return [], [], [], []

        x_list  = [self._x[k] for k in idx]
        y_list  = [self._y[k] for k in idx]
        have_yv = self._expect_yvar_for_all is True and (len(self._yvar) == len(self._x))
        yv_list = [self._yvar[k] for k in idx] if have_yv else []

        if not stack:
            return x_list, y_list, (yv_list if have_yv else []), idx

        x_s = torch.stack(x_list, dim=0)                     # (m, x_dim)
        y_s = torch.stack(y_list, dim=0)                     # (m, J)
        yv_s = torch.stack(yv_list, dim=0) if have_yv else None  # (m, J) or None
        return x_s, y_s, yv_s, idx
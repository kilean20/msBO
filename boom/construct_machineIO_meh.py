import os
import sys
import time
import datetime
import random
import warnings
import numpy as np
import logging
# import torch
import pandas as pd
import concurrent
from typing import Optional, List, Union, Dict
from copy import deepcopy as copy
from .utils import suppress_outputs

try:
    from IPython.display import display
except ImportError:
    display = print

DEFAULT_use_epics = False
DEFAULT_set_manually = True

try:
    from epics import caget as epics_caget
    from epics import caput as epics_caput
    from epics import caget_many as epics_caget_many
    from epics import caput_many as epics_caput_many
    epics_imported = True
    with suppress_outputs():
        if epics_caget("REA_EXP:ELMT") is not None:
            DEFAULT_isOK_PVs = None  # Skip check if machine is REA
            DEFAULT_isOK_vals = None
        else:
            DEFAULT_isOK_PVs = ["ACS_DIAG:CHP:STATE_RD"]   # is FRIB chopper on?
            DEFAULT_isOK_vals = [3]   # ACS_DIAG:CHP:STATE_RD=3 when FRIB chopper on
except ImportError:
    logging.warning("Failed to import 'epics'")
    epics_imported = False
    DEFAULT_isOK_PVs = None
    DEFAULT_isOK_vals = None

try:
    from phantasy import fetch_data as phantasy_fetch_data_orig
    from phantasy import ensure_set as phantasy_ensure_set_orig
    from phantasy import caget as phantasy_caget
    from phantasy import caput as phantasy_caput
    phantasy_imported = True
except ImportError:
    logging.warning("Failed to import 'phantasy'")
    phantasy_imported = False


if phantasy_imported:
    def phantasy_fetch_data(pvlist: List[str],
                            time_span: float = 1.0,
                            sample_interval: float = 0.2,
                            fillna_method: str = 'ffill', #'nearest',
                            **kws,
                            ):
        ave, df = phantasy_fetch_data_orig(pvlist, time_span=time_span, with_data=True, 
                                        data_opt={'with_timestamp': True,'fillna_method': fillna_method})
        resample_interval  = str(int(1000*sample_interval))+'ms'
        df = df.resample(resample_interval).mean().interpolate(method='linear',limit_direction='both')

        return ave,df

    def phantasy_ensure_set(setpoint_pv: List[str], 
                            readback_pv: List[str], 
                            goal: List[float], 
                            tol: List[float], 
                            timeout: float = 30.0, 
                            extra_monitors: List[str] = None, 
                            fillna_method: str = 'linear',   # linear reccommned as it is ramping data
                            **kws,
                            ):
        ret, df = phantasy_ensure_set_orig(setpoint_pv, readback_pv, goal, 
                                        tol=tol, timeout=timeout, extra_monitors=extra_monitors,
                                        keep_data=True, fillna_method = fillna_method)
        return ret, df.interpolate(method='linear',limit_direction='both')


if epics_imported:
    def epics_fetch_data(
        pvlist: List[str], 
        time_span: float = 5,   
        sample_interval: float = 0.2,
        **kws,
        ):
        t0 = time.monotonic()
        index = [datetime.datetime.now()]
        data = [epics_caget_many(pvlist)]
        while time.monotonic()-t0 < time_span:
            time.sleep(sample_interval)
            index.append(datetime.datetime.now())
            data.append(epics_caget_many(pvlist))
        df = pd.DataFrame(data,index=index,columns=pvlist)
        return df.mean().values, df

    def epics_ensure_set(setpoint_pv: List[str], 
                         readback_pv: List[str], 
                         goal: List[float], 
                         tol: List[float], 
                         timeout: float = 30.0, 
                         extra_monitors: List[str] = None, 
                         sample_interval: float = 0.2,
                         **kws,
                         ):
        t0 = time.monotonic()
        epics_caput_many(setpoint_pv,goal)
        tol = np.array(tol)
        goal = np.array(goal)
        extra_monitors = extra_monitors if extra_monitors is not None else []
        pvlist = setpoint_pv + readback_pv + extra_monitors
        val = epics_caget_many(pvlist)
        index = [datetime.datetime.now()]
        data = [val]
        while time.monotonic()-t0 < timeout and np.any(np.abs(val-goal)>tol):
            time.sleep(sample_interval)
            val = epics_caget_many(pvlist)
            index.append(datetime.datetime.now())
            data.append(val)
        df = pd.DataFrame(data,index=index,columns=pvlist)
        ret = 'PutFinish' if time.monotonic()-t0 < timeout else 'Timeout'
        return ret, df

class fetch_data_with_isOK:
    def __init__(self,
                 fetch_data_base, 
                 isOK_PVs  = DEFAULT_isOK_PVs, 
                 isOK_vals = DEFAULT_isOK_vals):
        self.fetch_data_base = fetch_data_base
        assert isinstance(isOK_PVs, list), f"Expected isOK_PVs to be of type list, but got {type(isOK_PVs).__name__}"
        self.isOK_PVs = isOK_PVs
        self.isOK_vals = np.array(isOK_vals) if isinstance(isOK_vals,list) else isOK_vals

    def __call__(self,pvlist,time_span,sample_interval=0.2):
        pvlist_expanded = pvlist + [pv for pv in self.isOK_PVs if pv not in pvlist]
        ave,df = self.fetch_data_base(pvlist_expanded,time_span,sample_interval=sample_interval)
        display(df) # debug
        while np.any(df[self.isOK_PVs].mean().values != self.isOK_vals):
            logging.warning(f"notOK from {self.isOK_PVs} detected during fetch_data. Re-try in 5 sec... ")
            logging.debug(df)
            time.sleep(5)
            ave,df = self.fetch_data_base(pvlist_expanded,time_span,sample_interval=sample_interval)
        return df[pvlist].mean().values, df[pvlist]


class construct_machineIO:
    def __init__(self,
                 ensure_set_timeout: int = 20, 
                 fetch_data_time_span: float = 2.0,
                 sample_interval: float = 0.2,
                 use_epics = DEFAULT_use_epics,
                 isOK_PVs  = DEFAULT_isOK_PVs,
                 isOK_vals = DEFAULT_isOK_vals,
                 set_manually = DEFAULT_set_manually,
                 virtual_machineIO = None,
                ):
        self.ensure_set_timeout = ensure_set_timeout
        self.ensure_set_timewait_after_ramp = 0.25
        self.fetch_data_time_span = fetch_data_time_span
        self.sample_interval = sample_interval
        self.isOK_PVs = isOK_PVs
        self.isOK_vals = isOK_vals
        self.set_manually = set_manually

        if virtual_machineIO:
            self.isOK_PVs = None
            self.isOK_vals = None
            self._ensure_set = virtual_machineIO.ensure_set
            self._fetch_data = virtual_machineIO.fetch_data
        elif use_epics:
            self._ensure_set = epics_ensure_set
            self._fetch_data = epics_fetch_data
            self._caget = epics_caget
            self._caput = epics_caput
        else:
            self._ensure_set = phantasy_ensure_set
            self._fetch_data = phantasy_fetch_data
            self._caget = phantasy_caget
            self._caput = phantasy_caput
        if self.isOK_PVs is not None:
            self._fetch_data = fetch_data_with_isOK(
                                    self._fetch_data,
                                    isOK_PVs = self.isOK_PVs,
                                    isOK_vals = self.isOK_vals)
        self.clear_history()
                                    
    def clear_history(self):
        self.history = {'set_order':None,
                        'df':None}
        
    def ensure_set(self,
                   setpoint_pv: List[str], 
                   readback_pv: List[str], 
                   goal: List[float], 
                   tol: List[float],
                   timeout: Union[int, None] = None,
                   sample_interval: float = None,
                   extra_monitors: Optional[List[str]] = None,                   
                   ):
        timeout = timeout or self.ensure_set_timeout
        sample_interval = sample_interval or self.sample_interval

        set_order = pd.DataFrame([goal],index=[datetime.datetime.now()],columns=setpoint_pv)
        if self.history['set_order'] is None:
            self.history['set_order'] = set_order
        else:
            self.history['set_order'] = pd.concat([self.history['set_order'],set_order])

        if self.set_manually:
            display(pd.DataFrame(goal,index=setpoint_pv).T)
            if isinstance(goal,np.ndarray):
                goal = goal.tolist()
            if isinstance(tol,np.ndarray):
                tol = tol.tolist()
            print(f"ensure_set({setpoint_pv},{readback_pv},{goal},tol={tol},timeout={timeout})") 
            input("Set the above PVs and press any key to continue...")
            ret, df = 'set_manually', None
        else:
            ret, df = self._ensure_set(setpoint_pv,readback_pv,goal,tol,
                                       timeout=timeout, sample_interval=sample_interval,
                                       extra_monitors = extra_monitors)
        if df is not None:
            if self.history['df'] is None:
                self.history['df'] = df
            else:
                self.history['df'] = pd.concat([self.history['df'], df], axis=0)
        time.sleep(self.ensure_set_timewait_after_ramp)
        return ret, df
    
    def fetch_data(self,
                   pvlist: List[str],
                   time_span: float = None, 
                   sample_interval : float = None,
                   ):

        time_span = time_span or self.fetch_data_time_span
        sample_interval = sample_interval or self.sample_interval

        ave, df = self._fetch_data(pvlist,
                                     time_span = time_span, 
                                     sample_interval = sample_interval,
                                     )                                     
        if self.history['df'] is None:
            self.history['df'] = df
        else:
            self.history['df'] = pd.concat([self.history['df'],df])
        return ave, df
    
    
def get_tolerance(PV_CSETs: List[str], machineIO: construct_machineIO):
    '''
    Automatically define tolerance
    tol is defined by 10% of ramping rate: i.e.) tol = ramping distance in a 0.1 sec
    PV_CSETs: list of CSET-PVs 
    '''
    pv_ramp_rate = []
    for pv in PV_CSETs:
        if 'PSOL' in pv:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RRTE_RSET')
        else:
            pv_ramp_rate.append(pv[:pv.rfind(':')]+':RSSV_RSET')
    try:
        ramp_rate,_ = machineIO._fetch_data(pv_ramp_rate,0.1)
        tol = 0.1*ramp_rate
    except:
        raise RuntimeError('tolerance for ramping could not be automatically determined. Please adjust it manually ')
    return tol
    
    
def get_limits(PV_CSETs: List[str], machineIO: construct_machineIO):
    '''
    Automatically retrive limit for PV put
    PV_CSETs: list of CSET-PVs 
    '''
    lo_lim = []
    hi_lim = []
    for pv in PV_CSETs:
        if ':V_CSET' in pv:
#             tmp = [pv.replace(':V_CSET',':V_CSET.LOPR'), pv.replace(':V_CSET',':V_CSET.HOPR')]
            tmp = [pv.replace(':V_CSET',':V_CSET.DRVL'), pv.replace(':V_CSET',':V_CSET.DRVH')]
            tmp,_ = machineIO._fetch_data(tmp,0.1)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        elif ':I_CSET' in pv:
#             tmp = [pv.replace(':I_CSET',':I_CSET.LOPR'), pv.replace(':I_CSET',':I_CSET.HOPR')]
            tmp = [pv.replace(':I_CSET',':I_CSET.DRVL'), pv.replace(':I_CSET',':I_CSET.DRVH')]
            tmp,_ =machineIO._fetch_data(tmp,0.1)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        else:
            raise RuntimeError(f'failed to find operation limit for {pv}. Manually ensure the control limit')
    lo_lim = np.array(lo_lim)
    hi_lim = np.array(hi_lim)
    assert np.all(lo_lim < hi_lim)
    return lo_lim, hi_lim



def get_RDs(PV_CSETs: List[str], machineIO: construct_machineIO):
    PV_RDs = []
    for pv in PV_CSETs:
        if '_CSET' in pv:
            PV_RDs.append(pv.replace('_CSET','_RD'))
        elif '_MTR.VAL' in pv:
            PV_RDs.append(pv.replace('_MTR.VAL','_MTR.RBV'))
        else:
            raise RuntimeError(f"Automatic decision of 'RD' for {PV_CSETs} failed")
    try:
        _,_ = machineIO._fetch_data(PV_RDs,0.1)
    except:
        raise RuntimeError(f"Automatic decision of 'RD' for {PV_CSETs} failed")
    return PV_RDs



class add_column_to_df:
    def __init__(self, 
        input_column_names: Optional[List[str]] = None,
        output_column_names: Optional[List[str]] = None,
        func: Optional[callable] = None,
        ):
        if input_column_names is None:
            input_column_names = []
        if output_column_names is None:
            output_column_names = []
        self.input_column_names = list(input_column_names)
        self.output_column_names = list(output_column_names)
        self.func = func
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.func is None:
            return df
        x = df[self.input_column_names].to_numpy()
        df[self.output_column_names] = self.func(x)        
        # # Validate output shape
        # if y.shape[0] != df.shape[0]:
        #     raise ValueError(f"Output shape {y.shape} does not match input shape {df.shape[0]}")        
        # if y.shape[1] != len(self.output_column_names):
        #     raise ValueError(f"Expected {len(self.output_column_names)} output columns but got {y.shape[1]}")
        return df


class Evaluator:
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs: List[str],
                 control_tols: Union[List[float], np.ndarray],
                 monitor_PVs: Optional[List[str]] = None,
                 control_couplings: Optional[dict] = None,
                 column_adders: Optional[List[add_column_to_df]] = None):
        """
        Initialize the evaluator with machine I/O and relevant data sets.
        
        Parameters:
        - machineIO: An instance of the machine I/O class.
        - control_CSETs: List of control PVs.
        - control_RDs: List of readback PVs.
        - control_tols: List or array of tolerances for each control PV.
        - monitor_PVs: Optional list of additional PVs to monitor.
        - control_couplings: CSETs that need to be coupled with one of control_CSETs. 
        """
        # Validate all input parameters
        self.machineIO = self._validate_machineIO(machineIO)
        self.control_CSETs = self._validate_control_CSETs(control_CSETs)
        self.control_RDs = self._validate_control_RDs(control_RDs, control_CSETs)
        self.control_tols = self._validate_control_tols(control_tols, control_CSETs)
        self.monitor_PVs = self._validate_monitor_PVs(monitor_PVs, control_CSETs, control_RDs)
        # print("Evaluator_init: control_couplings",control_couplings)
        self.control_couplings = self._validate_control_couplings(control_couplings, control_CSETs, control_RDs)
        # print("Evaluator_init:self.control_couplings",self.control_couplings)
        # Precompute expanded controls and coupling indices
        self.expanded_control_CSETs, self.expanded_control_RDs, self.expanded_control_tols, self.coupling_indices = self._precompute_control_couplings_and_indices()

        self.fetch_data_monitors = self.expanded_control_CSETs + self.expanded_control_RDs + self.monitor_PVs
        self.ensure_set_monitors = [
            m for m in self.fetch_data_monitors
            if m not in self.expanded_control_RDs and m not in self.expanded_control_CSETs
        ]

        if column_adders is None:
            column_adders = []
        self.column_adders = column_adders
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _validate_machineIO(self, machineIO):
        if not hasattr(machineIO, 'ensure_set') or not callable(getattr(machineIO, 'ensure_set', None)):
            raise TypeError("machineIO must have a callable `ensure_set` method.")
        if not hasattr(machineIO, 'fetch_data') or not callable(getattr(machineIO, 'fetch_data', None)):
            raise TypeError("machineIO must have a callable `fetch_data` method.")
        if not hasattr(machineIO, 'history'):
            raise AttributeError("machineIO must have a `history` attribute.")
        return machineIO

    def _validate_control_CSETs(self, control_CSETs):
        """Validate control_CSETs."""
        if not isinstance(control_CSETs, list) or not all(isinstance(cset, str) for cset in control_CSETs):
            raise TypeError("control_CSETs must be a list of strings.")
        if len(control_CSETs) != len(set(control_CSETs)):
            raise ValueError("control_CSETs contains duplicate entries.")
        return control_CSETs

    def _validate_control_RDs(self, control_RDs, control_CSETs):
        """Validate control_RDs."""
        if not isinstance(control_RDs, list) or not all(isinstance(rd, str) for rd in control_RDs):
            raise TypeError("control_RDs must be a list of strings.")
        if len(control_RDs) != len(set(control_RDs)):
            raise ValueError("control_RDs contains duplicate entries.")
        if len(control_RDs) != len(control_CSETs):
            raise ValueError("The length of control_RDs must match the length of control_CSETs.")
        return control_RDs

    def _validate_control_tols(self, control_tols, control_CSETs):
        """Validate control_tols."""
        if not isinstance(control_tols, (list, np.ndarray)) or not all(isinstance(tol, (int, float)) for tol in control_tols):
            raise TypeError("control_tols must be a list or numpy array of numbers.")
        if len(control_tols) != len(control_CSETs):
            raise ValueError("Length of control_tols must match length of control_CSETs.")
        return control_tols

    def _validate_monitor_PVs(self, monitor_PVs, control_CSETs, control_RDs):
        """Validate monitor_PVs."""
        if monitor_PVs is None:
            return []
        if not isinstance(monitor_PVs, list) or not all(isinstance(pv, str) for pv in monitor_PVs):
            raise TypeError("monitor_PVs must be a list of strings.")
        if len(monitor_PVs) != len(set(monitor_PVs)):
            raise ValueError("monitor_PVs contains duplicate entries.")
        if set(monitor_PVs).intersection(set(control_CSETs)):
            raise ValueError("monitor_PVs must be disjoint from control_CSETs.")
        if set(monitor_PVs).intersection(set(control_RDs)):
            raise ValueError("monitor_PVs must be disjoint from control_RDs.")
        return monitor_PVs

    def _validate_control_couplings(self, control_couplings, control_CSETs, control_RDs):
        """Validate control_couplings."""
        if control_couplings is None:
            return {}

        if not isinstance(control_couplings, dict):
            raise TypeError("control_couplings must be a dictionary.")

        coupling_indices = {}

        for key, value in control_couplings.items():
            if not isinstance(key, str) or key not in control_CSETs:
                raise ValueError(f"Key '{key}' in control_couplings must be a string and in control_CSETs.")
            if not isinstance(value, dict):
                raise TypeError(f"Value for '{key}' must be a dictionary.")

            required_keys = {"CSETs", "RDs", "coeffs", "tols"}
            if not required_keys.issubset(value.keys()):
                missing = required_keys - set(value.keys())
                raise ValueError(f"Value for '{key}' must contain keys: {required_keys}. Missing: {missing}")

            csets = value["CSETs"]
            rds = value["RDs"]
            coeff = value["coeffs"]
            tol = value["tols"]

            if not isinstance(csets, list) or not all(isinstance(c, str) for c in csets):
                raise TypeError(f"'CSETs' for '{key}' must be a list of strings.")
            if not isinstance(rds, list) or not all(isinstance(r, str) for r in rds):
                raise TypeError(f"'RDs' for '{key}' must be a list of strings.")
            if not isinstance(coeff, (list, np.ndarray)) or not all(isinstance(c, (int, float)) for c in coeff):
                raise TypeError(f"'coeffs' for '{key}' must be a list or numpy array of numbers.")
            if not isinstance(tol, (list, np.ndarray)) or not all(isinstance(t, (int, float)) for t in tol):
                raise TypeError(f"'tols' for '{key}' must be a list or numpy array of numbers.")

            if len(coeff) != len(csets) or len(tol) != len(csets) or len(rds) != len(csets):
                raise ValueError(f"Lengths of 'coeffs', 'tols', and 'RDs' must match the length of 'CSETs' for '{key}'.")
                
        return control_couplings

    def _precompute_control_couplings_and_indices(self):
        """
        Precomputes the expanded control_CSETs, control_RDs, and control_tols
        by applying control couplings during initialization, and also precomputes
        the indices for control couplings to optimize runtime performance.

        Returns:
        - A tuple containing:
          1. Expanded control_CSETs
          2. Expanded control_RDs
          3. Expanded control_tols
          4. A dictionary of coupling indices with associated coefficients
        """
        expanded_control_CSETs = list(self.control_CSETs)  # Create mutable copies
        expanded_control_RDs = list(self.control_RDs)
        expanded_control_tols = list(self.control_tols)
        coupling_indices = {}

        if self.control_couplings:
            for pv, value in self.control_couplings.items():
                # Extend control CSETs, RDs, and tols with coupled values
                expanded_control_CSETs.extend(value["CSETs"])
                expanded_control_RDs.extend(value["RDs"])
                expanded_control_tols.extend(value["tols"])

                # Precompute the coupling indices for runtime
                ipv = self.control_CSETs.index(pv)
                coupling_indices[pv] = {
                    "index": ipv,
                    "coeffs": np.array(value["coeffs"])
                }

        return expanded_control_CSETs, expanded_control_RDs, expanded_control_tols, coupling_indices

    def _apply_control_couplings_runtime(self, x):
        """
        Optimized application of runtime scaling to the input array based on precomputed coupling indices.

        Parameters:
        - x: Input array for initial control_CSETs.

        Returns:
        - Expanded x array after applying control couplings.
        """
        new_x_values = []
        for pv, data in self.coupling_indices.items():
            # print('_apply_control_couplings_runtime: data["index"],x[data["index"]]',data["index"],x[data["index"]])
            new_x_values.extend(data["coeffs"] * x[data["index"]])
        return np.concatenate([x, new_x_values])

    def set(self, x):
        """
        Sets the machine state and returns the result.

        Parameters:
        - x: Input array corresponding to initial control_CSETs.

        Returns:
        - Tuple containing a success flag (bool) and a DataFrame of ramping information.
        """
        x = np.asarray(x).flatten()
#         print("set: x",x)
#         print("set: self.control_CSETs",self.control_CSETs)
        assert len(x) == len(self.control_CSETs), "Length of input x must match the number of control_CSETs."
#         print("set: self.control_couplings",self.control_couplings)
#         print("set: self.expanded_control_CSETs",self.expanded_control_CSETs)
        # Apply runtime control couplings to expand x
        if self.control_couplings:
            x = self._apply_control_couplings_runtime(x)

        # Ensure the machine IO is set
        ret, ramping_df = self.machineIO.ensure_set(
            self.expanded_control_CSETs,
            self.expanded_control_RDs,
            x,
            self.expanded_control_tols,
            extra_monitors=self.ensure_set_monitors
        )
        #print("set: ensure_set done")
        for adder in self.column_adders:
            ramping_df = adder(ramping_df)
        return ret, ramping_df

    def read(self):
        """
        Reads the current data from the machine.

        Returns:
        - DataFrame containing the fetched data.
        """
        #print("read: self.fetch_data_monitors",self.fetch_data_monitors)
        _, df = self.machineIO.fetch_data(self.fetch_data_monitors)
        for adder in self.column_adders:
            df = adder(df)
        return df

    def set_n_read(self, x):
        """
        Combines set and read operations.

        Parameters:
        - x: Input array corresponding to initial control_CSETs.

        Returns:
        - Tuple containing the fetched DataFrame and ramping information.
        """
        ret, ramping_df = self.set(x)
        df = self.read()
        return df, ramping_df

    def submit(self, x):
        """
        Submit a task to set and read data asynchronously.

        Parameters:
        - x: Input array corresponding to initial control_CSETs.

        Returns:
        - Future object for the asynchronous task.
        """
        future = self.executor.submit(self.set_n_read, x)
        return future

    def get_result(self, future: concurrent.futures.Future):
        """
        Retrieves the result from an asynchronous task.

        Parameters:
        - future: Future object returned by submit.

        Returns:
        - Result of the task, or raises an error if the task failed.
        """
        try:
            return future.result()
        except Exception as e:
            raise RuntimeError(f"Error occurred while processing future: {e}")


class MultiStateEvaluator(Evaluator):
    def __init__(self,
                 machineIO,
                 control_CSETs: List[str],
                 control_RDs  : List[str],
                 control_tols : Union[List[float], np.ndarray],
                 state_CSETs : List[str],
                 state_RDs   : List[str],
                 state_tols  : List[str],
                 state_vals  : Union[List[float], np.ndarray],
                 control_couplings: Optional[dict] = None,
                 state_names : Optional[List[str]] = None,
                 monitor_PVs : Optional[List[str]] = None,
                 column_adders: Optional[List[add_column_to_df]] = None,
                 ):
    
        self._validate_state_inputs(state_CSETs, state_RDs, state_tols, state_vals, state_names)

        if state_names is None:
            state_names = ['state'+str(i) for i in range(len(state_vals))]
        else:
            assert len(state_names) == len(state_vals)

        self.state_CSETs  = state_CSETs
        self.state_RDs    = state_RDs
        self.state_tols   = state_tols
        self.state_vals   = np.array(state_vals)
        self.state_names  = state_names

        super().__init__(                
            machineIO = machineIO,
            control_CSETs = control_CSETs + state_CSETs,
            control_RDs   = control_RDs + state_RDs,
            control_tols  = np.concatenate((control_tols, state_tols)),
            control_couplings = control_couplings,
            monitor_PVs = monitor_PVs,
            column_adders = column_adders,
            )

    def _validate_state_inputs(self, state_CSETs, state_RDs, state_tols, state_vals, state_names):
            """Consolidated validation function for all parameters."""
    
            assert isinstance(state_CSETs, list), f"Expected state_CSETs to be of type list, but got {type(state_CSETs).__name__}"
            assert isinstance(state_RDs, list), f"Expected state_RDs to be of type list, but got {type(state_RDs).__name__}"
            assert isinstance(state_tols, list), f"Expected state_tols to be of type list but got {type(state_tols).__name__}"
            assert isinstance(state_vals, (list, np.ndarray)), f"Expected state_vals to be of type list or np.ndarray, but got {type(state_vals).__name__}"
            state_vals = np.array(state_vals)
            assert state_vals.ndim == 2, "state_vals must be a 2D array"
            assert state_vals.shape[1] == len(state_CSETs), "state_vals must be a 2D array with the same number of columns as state_CSETs"
            
            assert len(state_CSETs) == len(set(state_CSETs)), "state_CSETs contains duplicates"
            assert len(state_RDs) == len(set(state_RDs)), "state_RDs contains duplicates"
            
            assert set(state_CSETs).isdisjoint(set(state_RDs)), "state_CSETs and state_RDs should be disjoint."
    
            if state_names is not None:
                assert isinstance(state_names, list), f"Expected state_names to be of type list, but got {type(state_names).__name__}"
                assert len(state_names) == state_vals.shape[0], "Length of state_names must match the number of rows in state_vals."
    

    def add_state_column_to_df(self, df):
        if df is None:
            return None
        reads = df[self.state_CSETs].values[0]
#         print("add_state_column_to_df: self.state_CSETs, reads",self.state_CSETs, reads)
#         print("add_state_column_to_df: self.state_names",self.state_names)
        state = None
        for i, vals in enumerate(self.state_vals):
            if np.allclose(reads, vals, atol=1e-6):
                state = self.state_names[i]
                break
        df["state"] = state
        return df
                
    def read(self):
        return self.add_state_column_to_df(super().read())

        
    def set(self, x, istate):
        # if isinstance(x, torch.tensor):
        #     x = x.detach().cpu().numpy()
        x_with_state = np.concatenate((x, self.state_vals[istate]))
        
        ret, ramping_df = super().set(x_with_state)
        # ret, ramping_df = self.machineIO.ensure_set(self.control_CSETs, 
        #                                             self.control_RDs, 
        #                                             x_with_state,
        #                                             self.control_tols,
        #                                             extra_monitors=self.ensure_set_monitors
        #                                             )
        
#         print("MultiStateEvaluator:set: ramping_df",ramping_df)
        
        return ret, self.add_state_column_to_df(ramping_df)
        # return ret, ramping_df

    def set_n_read(self,x,istate):
#         print("MultiStateEvaluator:set_n_read: x,istate",x,istate)
        ret, ramping_df = self.set(x,istate)
#         print("MultiStateEvaluator:set_n_read: ret, ramping_df",ret, ramping_df)
        df = self.read()
#         print("MultiStateEvaluator:set_n_read: df",df)
        return df, ramping_df
        
    def submit(self, x, istate):
        """
        Submit a task to set and read data asynchronously.
        """
        future = self.executor.submit(self.set_n_read, x, istate)
        return future

    def get_result(self, future: concurrent.futures.Future):
        try:
            return future.result()
        except Exception as e:
            raise RuntimeError(f"Error occurred while processing future: {e}")
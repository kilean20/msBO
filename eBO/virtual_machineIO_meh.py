import numpy as np
import pandas as pd
import datetime
from typing import List, Optional

class VirtualMachineIO:
    def __init__(self):
        self.trian_x_set = pd.DataFrame()
        self.trian_x_rd  = pd.DataFrame()  # Initialize as DataFrame instead of empty list
        self.trian_y     = pd.DataFrame()  # Initialize as DataFrame for consistency
        self.model = None  # Placeholder for a model object
        self.i_data = 0
        self.input_dim = 0
        self.output_dim = 0

    def ensure_set(self, setpoint_pv: List[str], readback_pv: List[str], goal: np.ndarray, tol: float, timeout: float, sample_interval: Optional[float] = None):
        """Ensure set data for setpoint and readback PVs are updated."""
        if isinstance(goal,list):
            goal = np.array()
        # Create DataFrame for setpoint_pv and readback_pv
        df_setpoint = pd.DataFrame([goal], index=[self.i_data], columns=setpoint_pv)
        df_readback = pd.DataFrame([goal + 1e-6*np.random.randn(len(goal))], index=[self.i_data], columns=readback_pv)
        # Update set_data by concatenating new data with existing set_data
        if self.trian_x_set.empty:
            self.trian_x_set = df_setpoint
            self.trian_x_rd  = df_readback
        else:
            self.trian_x_set = pd.concat([self.trian_x_set, df_setpoint], axis=0)
            self.trian_x_rd  = pd.concat([self.trian_x_rd , df_readback], axis=0)
        self.trian_x_set.fillna(method='bfill', inplace=True)
        self.trian_x_rd .fillna(method='bfill', inplace=True)

        self.i_data, self.input_dim = self.trian_x_rd.shape

        return 'Put Finished', None

    def fetch_data(self, pvlist: List[str], time_span: Optional[float] = None, sample_interval: Optional[float] = None):
        """Fetch or simulate data for the given PVs."""
        # If there is no read_data, or if it's smaller than expected, generate random values
        if len(self.trian_y) <= 2*input_dim:
            vals = np.random.rand(len(pvlist))
            df = pd.DataFrame([vals], index=[self.i_data], columns=setpoint_pv)
            # if self.trian_y.empty:
                # vals = np.random.rand(len(pvlist))
                # self.trian_y = pd.DataFrame([vals], index=[self.i_data], columns=setpoint_pv)
            # else:
                # if self.i_data in self.train_y.index:
                    # if set(pvlist) is not subset(self.trian_y.columns):
                        # cols = [pv for pv in setpoint_pv if pv not in self.train_y.columns]
                        # vals = np.random.rand(len(cols))
                        # df_fetch = pd.DataFrame([vals], index=[self.i_data], columns=cols)
                # else:    
        elif set(pvlist) is subset(self.trian_y.columns):
            pred_y = self.model(self.self.trian_x_rd.iloc[-1,:].values)
            
            
                cols = list(self.train_y.columns) + [pv for pv in setpoint_pv if pv not in self.train_y.columns]
                vals = np.random.rand(len(cols))
                df_fetch = pd.DataFrame([vals], index=[self.i_data], columns=setpoint_pv)
            
        else:
            # Check if all pvlist items are present in read_data
            if set(pvlist).issubset(set(self.read_data.columns)):
                # Use the model to predict values based on the last row of set_data
                vals = self.model(self.set_data.iloc[-1, :].values)
            else:
                # Add new PVs if they are not in read_data
                new_pv = [pv for pv in pvlist if pv not in self.read_data.columns]
                self.read_data = self.read_data.reindex(columns=self.read_data.columns.tolist() + new_pv)

                # Refit the model with updated set_data and read_data
                self.model = self.fit_model(self.set_data, self.read_data)

                # Predict using the updated model
                vals = self.model(self.set_data.iloc[-1, :].values)

        return vals

    def fit_model(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit a model to the given data (placeholder for actual model fitting)."""
        # Placeholder for model fitting logic
        # In a real scenario, you would implement model training here (e.g., a regression model)
        def dummy_model(input_data):
            # Dummy model that simply returns random values for now
            return np.random.rand(len(y.columns))

        return dummy_model

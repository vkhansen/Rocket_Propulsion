# model.py
import numpy as np
from constants import G0, EPSILON, PAYLOAD_FRACTION, GRAVITY_LOSS, DRAG_LOSS

class RocketModel:
    def __init__(self, isp, total_delta_v):
        """
        Initialize the rocket model with stage-specific parameters.
        """
        self.isp = np.array(isp)
        self.total_delta_v = total_delta_v

    def delta_v_function(self, stage_fractions):
        """
        Compute the engine-provided ΔV per stage.
        
        Parameters:
            stage_fractions (array-like): The design variables (mass fractions) for each stage.
            
        Returns:
            np.array: ΔV contribution per stage.
        """
        return G0 * self.isp * np.log(1 / (EPSILON + PAYLOAD_FRACTION + np.array(stage_fractions)))
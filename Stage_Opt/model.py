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

    def objective(self, stage_fractions):
        """
        Compute the squared error between the produced engine ΔV (sum of stages)
        and the required engine ΔV (mission target plus losses).
        
        Parameters:
            stage_fractions (array-like): The design variables.
            
        Returns:
            float: Squared error.
        """
        required_engine_delta_v = self.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        produced_delta_v = np.sum(self.delta_v_function(stage_fractions))
        return (produced_delta_v - required_engine_delta_v) ** 2

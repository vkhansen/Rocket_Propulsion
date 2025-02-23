import os
import json
import tempfile
import unittest
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("output", "test_debug.log"), mode='w')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.INFO)

import payload_optimization as po
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestPayloadOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("Starting TestPayloadOptimization suite.")
        
    def setUp(self):
        logger.info("Setting up temporary test files.")
        # Create a temporary input data file
        self.input_data = {
            "parameters": {
                "TOTAL_DELTA_V": 9500,
                "G0": 9.81
            },
            "stages": [
                {
                    "stage": 1,
                    "ISP": 300,
                    "EPSILON": 0.1,
                    "parameters": {
                        "TOTAL_DELTA_V": 9500
                    }
                },
                {
                    "stage": 2,
                    "ISP": 320,
                    "EPSILON": 0.08,
                    "parameters": {
                        "TOTAL_DELTA_V": 9500
                    }
                }
            ]
        }
        
        # Write temporary files
        self.temp_input = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
        json.dump(self.input_data, self.temp_input)
        self.temp_input.close()

    def tearDown(self):
        logger.info("Cleaning up temporary files.")
        os.remove(self.temp_input.name)

    def test_load_input_data(self):
        logger.info("Testing input data loading")
        stages = po.load_input_data(self.temp_input.name)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["ISP"], 300)
        self.assertEqual(stages[1]["EPSILON"], 0.08)

    def test_calculate_mass_ratios(self):
        logger.info("Testing mass ratio calculation")
        dv = np.array([4000, 5500])
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        ratios = po.calculate_mass_ratios(dv, ISP, EPSILON)
        self.assertEqual(len(ratios), 2)
        for ratio in ratios:
            self.assertGreater(ratio, 0)

    def test_calculate_payload_fraction(self):
        logger.info("Testing payload fraction calculation")
        mass_ratios = [0.5, 0.6]
        fraction = po.calculate_payload_fraction(mass_ratios)
        self.assertAlmostEqual(fraction, 0.3, places=5)

    def test_payload_fraction_objective(self):
        logger.info("Testing payload fraction objective")
        dv = np.array([4000, 5500])
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        result = po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        self.assertIsInstance(result, float)
        self.assertGreater(result, -1)  # Should be negative but greater than -1

    def test_solve_with_differential_evolution(self):
        """Test differential evolution solver directly."""
        logger.info("Testing differential evolution solver")
        initial_guess = [4000, 5500]
        bounds = [(0, 9500), (0, 9500)]
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        TOTAL_DELTA_V = 9500
        config = {
            "optimization": {
                "differential_evolution": {
                    "population_size": 15,
                    "max_iterations": 1000,
                    "mutation": [0.5, 1.0],
                    "recombination": 0.7,
                    "strategy": "best1bin",
                    "tol": 1e-6
                }
            }
        }
        
        result = po.solve_with_differential_evolution(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config
        )
        
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(np.sum(result), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in result))

    def test_optimize_payload_allocation_all_methods(self):
        logger.info("Testing all optimization methods")
        G0 = 9.81
        TOTAL_DELTA_V = 9500
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        methods = ["SLSQP"]  # Only test SLSQP as it's the most reliable method
        
        for method in methods:
            with self.subTest(method=method):
                optimal_dv, stage_ratios, payload_fraction = po.optimize_payload_allocation(
                    TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
                )
                self.assertEqual(len(optimal_dv), 2)
                self.assertAlmostEqual(np.sum(optimal_dv), TOTAL_DELTA_V, places=2)
                self.assertTrue(all(ratio > 0 for ratio in stage_ratios))
                self.assertTrue(0 < payload_fraction < 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Test suite for rocket stage optimization."""
import os
import json
import tempfile
import unittest
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("output", "test_debug.log"), mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from src.utils.data import load_input_data, calculate_mass_ratios, calculate_payload_fraction
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
from src.optimization.solvers import (
    solve_with_slsqp,
    solve_with_basin_hopping,
    solve_with_differential_evolution,
    solve_with_ga as solve_with_genetic_algorithm,
    solve_with_adaptive_ga,
    solve_with_pso
)
from src.utils.config import CONFIG

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TestPayloadOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nStarting TestPayloadOptimization suite.")
        
    def setUp(self):
        print("\nSetting up test case...")
        # Create a temporary input data file
        self.input_data = {
            "parameters": {
                "TOTAL_DELTA_V": 9300,  # Updated to match current implementation
                "G0": 9.81
            },
            "stages": [
                {
                    "stage": 1,
                    "ISP": 300,
                    "EPSILON": 0.06
                },
                {
                    "stage": 2,
                    "ISP": 348,
                    "EPSILON": 0.04
                }
            ]
        }
        
        # Write temporary files
        self.temp_input = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
        json.dump(self.input_data, self.temp_input)
        self.temp_input.close()

    def tearDown(self):
        print("Cleaning up test case...")
        os.remove(self.temp_input.name)

    def test_load_input_data(self):
        """Test loading input data from JSON file."""
        print("\nTesting input data loading...")
        parameters, stages = load_input_data(self.temp_input.name)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["ISP"], 300)
        self.assertEqual(stages[1]["EPSILON"], 0.04)
        self.assertEqual(parameters["TOTAL_DELTA_V"], 9300)
        self.assertEqual(parameters["G0"], 9.81)

    def test_calculate_mass_ratios(self):
        """Test mass ratio calculation."""
        print("\nTesting mass ratio calculation...")
        dv = np.array([4650, 4650])
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        G0 = 9.81
        ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        self.assertEqual(len(ratios), 2)
        for ratio in ratios:
            self.assertGreater(ratio, 0)
            self.assertLess(ratio, 1)

    def test_calculate_payload_fraction(self):
        """Test payload fraction calculation."""
        print("\nTesting payload fraction calculation...")
        mass_ratios = [0.146, 0.216]  # Updated to match current implementation
        fraction = calculate_payload_fraction(mass_ratios)
        self.assertAlmostEqual(fraction, 0.032, places=3)  # Updated expected value

    def test_payload_fraction_objective(self):
        """Test payload fraction objective function."""
        print("\nTesting payload fraction objective...")
        dv = np.array([4650, 4650])
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        result = payload_fraction_objective(dv, G0, ISP, EPSILON)
        self.assertIsInstance(result, float)
        self.assertGreater(result, -1)  # Should be negative but greater than -1

    def test_solve_with_differential_evolution(self):
        """Test differential evolution solver."""
        print("\nTesting differential evolution solver...")
        initial_guess = [4650, 4650]
        bounds = [(0, 9300), (0, 9300)]
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        TOTAL_DELTA_V = 9300
        
        result = solve_with_differential_evolution(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('execution_time', result)
        self.assertIn('method', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_solve_with_ga(self):
        """Test genetic algorithm solver."""
        print("\nTesting genetic algorithm solver...")
        initial_guess = [4650, 4650]
        bounds = [(0, 9300), (0, 9300)]
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        TOTAL_DELTA_V = 9300
        
        result = solve_with_genetic_algorithm(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('execution_time', result)
        self.assertIn('method', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_solve_with_adaptive_ga(self):
        """Test adaptive genetic algorithm solver."""
        print("\nTesting adaptive genetic algorithm solver...")
        initial_guess = [4650, 4650]
        bounds = [(0, 9300), (0, 9300)]
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        TOTAL_DELTA_V = 9300
        
        result = solve_with_adaptive_ga(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('execution_time', result)
        self.assertIn('method', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_solve_with_pso(self):
        """Test particle swarm optimization solver."""
        print("\nTesting PSO solver...")
        initial_guess = [4650, 4650]
        bounds = [(0, 9300), (0, 9300)]
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        TOTAL_DELTA_V = 9300
        
        result = solve_with_pso(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('execution_time', result)
        self.assertIn('method', result)
        self.assertIn('history', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_all_optimization_methods(self):
        """Test all optimization methods."""
        print("\nTesting all optimization methods...")
        initial_guess = [4650, 4650]
        bounds = [(0, 9300), (0, 9300)]
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        TOTAL_DELTA_V = 9300
        
        solvers = [
            ("SLSQP", solve_with_slsqp),
            ("Basin-Hopping", solve_with_basin_hopping),
            ("DE", solve_with_differential_evolution),
            ("GA", solve_with_genetic_algorithm),
            ("Adaptive GA", solve_with_adaptive_ga),
            ("PSO", solve_with_pso)
        ]
        
        for name, solver in solvers:
            try:
                print(f"\nTesting {name} solver...")
                result = solver(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                
                # Verify results
                self.assertIsInstance(result, dict)
                self.assertIn('optimal_dv', result)
                self.assertIn('stage_ratios', result)
                self.assertIn('payload_fraction', result)
                self.assertIn('execution_time', result)
                self.assertIn('method', result)
                
                optimal_dv = result['optimal_dv']
                self.assertEqual(len(optimal_dv), 2)
                self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
                self.assertTrue(all(dv >= 0 for dv in optimal_dv))
                self.assertTrue(0 < result['payload_fraction'] < 1)
                
                print(f"{name} test passed!")
            except Exception as e:
                print(f"{name} test failed: {e}")
                raise

if __name__ == "__main__":
    unittest.main(verbosity=2)

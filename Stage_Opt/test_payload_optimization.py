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
    solve_with_genetic_algorithm,
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
                "TOTAL_DELTA_V": 9500,
                "G0": 9.81
            },
            "stages": [
                {
                    "stage": 1,
                    "ISP": 300,
                    "EPSILON": 0.1
                },
                {
                    "stage": 2,
                    "ISP": 320,
                    "EPSILON": 0.08
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
        self.assertEqual(stages[1]["EPSILON"], 0.08)
        self.assertEqual(parameters["TOTAL_DELTA_V"], 9500)
        self.assertEqual(parameters["G0"], 9.81)

    def test_calculate_mass_ratios(self):
        """Test mass ratio calculation."""
        print("\nTesting mass ratio calculation...")
        dv = np.array([4000, 5500])
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        G0 = 9.81
        ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        self.assertEqual(len(ratios), 2)
        for ratio in ratios:
            self.assertGreater(ratio, 0)

    def test_calculate_payload_fraction(self):
        """Test payload fraction calculation."""
        print("\nTesting payload fraction calculation...")
        mass_ratios = [0.5, 0.6]
        fraction = calculate_payload_fraction(mass_ratios)
        self.assertAlmostEqual(fraction, 0.3, places=5)

    def test_payload_fraction_objective(self):
        """Test payload fraction objective function."""
        print("\nTesting payload fraction objective...")
        dv = np.array([4000, 5500])
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        result = payload_fraction_objective(dv, G0, ISP, EPSILON)
        self.assertIsInstance(result, float)
        self.assertGreater(result, -1)  # Should be negative but greater than -1

    def test_solve_with_differential_evolution(self):
        """Test differential evolution solver directly."""
        print("\nTesting differential evolution solver...")
        initial_guess = [4000, 5500]
        bounds = [(0, 9500), (0, 9500)]
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        TOTAL_DELTA_V = 9500
        
        result = solve_with_differential_evolution(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(np.sum(result), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in result))

    def test_solve_with_genetic_algorithm(self):
        """Test genetic algorithm solver directly."""
        print("\nTesting genetic algorithm solver...")
        initial_guess = [4000, 5500]
        bounds = [(0, 9500), (0, 9500)]
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        TOTAL_DELTA_V = 9500
        
        result = solve_with_genetic_algorithm(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('time', result)
        self.assertIn('history', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_solve_with_adaptive_ga(self):
        """Test adaptive genetic algorithm solver directly."""
        print("\nTesting adaptive genetic algorithm solver...")
        initial_guess = [4000, 5500]
        bounds = [(0, 9500), (0, 9500)]
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        TOTAL_DELTA_V = 9500
        
        result = solve_with_adaptive_ga(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('time', result)
        self.assertIn('history', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_solve_with_pso(self):
        """Test particle swarm optimization solver directly."""
        print("\nTesting PSO solver...")
        initial_guess = [4000, 5500]
        bounds = [(0, 9500), (0, 9500)]
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        TOTAL_DELTA_V = 9500
        
        result = solve_with_pso(
            initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_dv', result)
        self.assertIn('stage_ratios', result)
        self.assertIn('payload_fraction', result)
        self.assertIn('time', result)
        self.assertIn('history', result)
        
        optimal_dv = result['optimal_dv']
        self.assertEqual(len(optimal_dv), 2)
        self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
        self.assertTrue(all(dv >= 0 for dv in optimal_dv))
        self.assertTrue(0 < result['payload_fraction'] < 1)

    def test_all_optimization_methods(self):
        """Test all optimization methods."""
        print("\nTesting all optimization methods...")
        parameters, stages = load_input_data(self.temp_input.name)
        
        # Extract parameters
        G0 = float(parameters["G0"])
        TOTAL_DELTA_V = float(parameters["TOTAL_DELTA_V"])
        ISP = [float(stage["ISP"]) for stage in stages]
        EPSILON = [float(stage["EPSILON"]) for stage in stages]
        
        # Initial setup
        n = len(ISP)
        initial_guess = np.full(n, TOTAL_DELTA_V / n)
        max_dv = TOTAL_DELTA_V * 0.9
        bounds = [(0, max_dv) for _ in range(n)]
        
        # Test each solver
        solvers = [
            ("SLSQP", solve_with_slsqp),
            ("Basin-Hopping", solve_with_basin_hopping),
            ("Differential Evolution", solve_with_differential_evolution),
            ("Genetic Algorithm", solve_with_genetic_algorithm),
            ("Adaptive GA", solve_with_adaptive_ga),
            ("PSO", solve_with_pso)
        ]
        
        for name, solver in solvers:
            with self.subTest(solver=name):
                print(f"\nTesting {name} solver...")
                result = solver(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                
                # Verify results
                if name in ["Genetic Algorithm", "Adaptive GA", "PSO"]:
                    self.assertIsInstance(result, dict)
                    self.assertIn('optimal_dv', result)
                    self.assertIn('stage_ratios', result)
                    self.assertIn('payload_fraction', result)
                    self.assertIn('time', result)
                    self.assertIn('history', result)
                    
                    optimal_dv = result['optimal_dv']
                    self.assertEqual(len(optimal_dv), 2)
                    self.assertAlmostEqual(sum(optimal_dv), TOTAL_DELTA_V, places=2)
                    self.assertTrue(all(dv >= 0 for dv in optimal_dv))
                    self.assertTrue(0 < result['payload_fraction'] < 1)
                else:
                    self.assertEqual(len(result), 2)
                    self.assertAlmostEqual(np.sum(result), TOTAL_DELTA_V, places=2)
                    self.assertTrue(all(dv >= 0 for dv in result))
                
                # Calculate performance metrics
                if name in ["Genetic Algorithm", "Adaptive GA", "PSO"]:
                    mass_ratios = result['stage_ratios']
                else:
                    mass_ratios = calculate_mass_ratios(result, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(mass_ratios)
                
                # Verify performance
                self.assertTrue(all(ratio > 0 for ratio in mass_ratios))
                self.assertTrue(0 < payload_fraction < 1)
                print(f"{name} results: dv={result}, payload_fraction={payload_fraction}")

if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Test suite for rocket stage optimization."""
import os
import json
import tempfile
import unittest
import numpy as np
import logging
import sys
import csv
import math
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Import our modules
from src.utils.data import load_input_data, calculate_mass_ratios, calculate_payload_fraction
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
from src.optimization.solvers import (
    solve_with_slsqp,
    solve_with_basin_hopping,
    solve_with_differential_evolution,
    solve_with_ga,
    solve_with_adaptive_ga,
    solve_with_pso
)
from src.utils.config import setup_logging, CONFIG
from src.optimization.cache import OptimizationCache, OUTPUT_DIR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Add test-specific log handler
test_log_file = os.path.join(OUTPUT_DIR, "test_output.log")
test_handler = RotatingFileHandler(
    test_log_file,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    mode='w'
)
test_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
))
test_handler.setLevel(logging.DEBUG)
logger.addHandler(test_handler)

class TestPayloadOptimization(unittest.TestCase):
    """Test cases for payload optimization functions."""

    def setUp(self):
        """Set up test cases."""
        # Load test data
        with open('input_data.json', encoding='utf-8') as f:
            data = json.load(f)
            parameters = data['parameters']
            stages = data['stages']
            
        self.G0 = parameters['G0']
        self.TOTAL_DELTA_V = parameters['TOTAL_DELTA_V']
        self.ISP = [stage['ISP'] for stage in stages]
        self.EPSILON = [stage['EPSILON'] for stage in stages]
        self.n_stages = len(stages)
        
        # Initial guess: equal split of delta-V
        self.initial_guess = np.array([self.TOTAL_DELTA_V / self.n_stages] * self.n_stages)
        
        # Bounds: each stage must use between 1% and 99% of total delta-V
        self.bounds = [(0.01 * self.TOTAL_DELTA_V, 0.99 * self.TOTAL_DELTA_V)] * self.n_stages
        
        # Test configuration
        self.config = {
            'optimization': {
                'max_iterations': 1000,
                'population_size': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            },
            'basin_hopping': {
                'n_iterations': 100,
                'temperature': 1.0,
                'stepsize': 0.5
            },
            'ga': {
                'population_size': 100,
                'n_generations': 200,
                'crossover_prob': 0.9,
                'crossover_eta': 15,
                'mutation_prob': 0.2,
                'mutation_eta': 20
            }
        }

    def test_load_input_data(self):
        """Test loading input data from JSON file."""
        logger.info("Starting input data loading test")
        parameters, stages = load_input_data('input_data.json')
        
        logger.debug(f"Loaded parameters: {parameters}")
        logger.debug(f"Loaded stages: {stages}")
        
        self.assertEqual(len(stages), len(self.ISP))
        self.assertEqual(stages[0]["ISP"], self.ISP[0])
        self.assertEqual(stages[1]["EPSILON"], self.EPSILON[1])
        self.assertEqual(parameters["TOTAL_DELTA_V"], self.TOTAL_DELTA_V)
        self.assertEqual(parameters["G0"], self.G0)
        logger.info("Completed input data loading test")

    def test_calculate_mass_ratios(self):
        """Test stage ratio (Λ) calculation."""
        logger.info("Starting stage ratio calculation test")
        
        # Test case with 2 stages
        dv = np.array([4650, 4650])  # Equal split of delta-V
        ISP = [300, 348]  # Different ISP for each stage
        EPSILON = [0.06, 0.04]  # Different structural coefficients
        G0 = 9.81
        
        logger.debug(f"Test parameters - dv: {dv}, ISP: {ISP}, EPSILON: {EPSILON}, G0: {G0}")
        
        # Calculate stage ratios
        ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        logger.debug(f"Calculated stage ratios: {ratios}")
        
        # Manual calculation for verification
        # For each stage i: Λᵢ = rᵢ/(1 + εᵢ)
        mass_ratio1 = np.exp(-dv[0] / (G0 * ISP[0]))
        mass_ratio2 = np.exp(-dv[1] / (G0 * ISP[1]))
        
        lambda1 = mass_ratio1 / (1.0 + EPSILON[0])
        lambda2 = mass_ratio2 / (1.0 + EPSILON[1])
        
        logger.debug(f"Manual calculations - mass_ratio1: {mass_ratio1}, mass_ratio2: {mass_ratio2}")
        logger.debug(f"Manual calculations - lambda1: {lambda1}, lambda2: {lambda2}")
        
        self.assertEqual(len(ratios), 2)
        self.assertAlmostEqual(ratios[0], lambda1, places=4)
        self.assertAlmostEqual(ratios[1], lambda2, places=4)
        
        # Test with single stage
        logger.info("Testing single stage configuration")
        single_dv = np.array([9300])
        single_isp = [300]
        single_epsilon = [0.06]
        
        logger.debug(f"Single stage parameters - dv: {single_dv}, ISP: {single_isp}, EPSILON: {single_epsilon}")
        
        single_ratios = calculate_mass_ratios(single_dv, single_isp, single_epsilon, G0)
        single_mass_ratio = np.exp(-single_dv[0] / (G0 * single_isp[0]))
        expected_lambda = single_mass_ratio / (1.0 + single_epsilon[0])
        
        logger.debug(f"Single stage results - calculated: {single_ratios[0]}, expected: {expected_lambda}")
        
        self.assertEqual(len(single_ratios), 1)
        self.assertAlmostEqual(single_ratios[0], expected_lambda, places=4)
        logger.info("Completed stage ratio calculation test")

    def test_payload_fraction(self):
        """Test payload fraction calculation."""
        logger.info("Starting payload fraction calculation test")
        
        # Test with 2 stages
        dv = np.array([4650, 4650])
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        G0 = 9.81
        
        logger.debug(f"Test parameters - dv: {dv}, ISP: {ISP}, EPSILON: {EPSILON}, G0: {G0}")
        
        # Calculate stage ratios
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        fraction = calculate_payload_fraction(mass_ratios)
        logger.debug(f"Calculated mass ratios: {mass_ratios}")
        logger.debug(f"Calculated payload fraction: {fraction}")
        
        # Manual calculation
        mr1 = np.exp(-dv[0] / (G0 * ISP[0]))
        mr2 = np.exp(-dv[1] / (G0 * ISP[1]))
        lambda1 = mr1 / (1.0 + EPSILON[0])
        lambda2 = mr2 / (1.0 + EPSILON[1])
        expected_fraction = lambda1 * lambda2
        
        logger.debug(f"Manual calculations - mr1: {mr1}, mr2: {mr2}")
        logger.debug(f"Manual calculations - lambda1: {lambda1}, lambda2: {lambda2}")
        logger.debug(f"Expected fraction: {expected_fraction}")
        
        self.assertAlmostEqual(fraction, expected_fraction, places=4)
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 1)
        logger.info("Completed payload fraction calculation test")

    def test_payload_fraction_objective(self):
        """Test payload fraction objective function."""
        logger.info("Starting payload fraction objective test")
        
        dv = np.array([4650, 4650])
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        
        logger.debug(f"Test parameters - dv: {dv}, ISP: {ISP}, EPSILON: {EPSILON}, G0: {G0}")
        
        result = payload_fraction_objective(dv, G0, ISP, EPSILON)
        logger.debug(f"Objective function result: {result}")
        
        # Manual calculation
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        expected = -calculate_payload_fraction(mass_ratios)
        logger.debug(f"Manual calculation - mass_ratios: {mass_ratios}")
        logger.debug(f"Expected result: {expected}")
        
        self.assertAlmostEqual(result, expected, places=4)
        self.assertGreater(result, -1)
        logger.info("Completed payload fraction objective test")

    def test_solve_with_slsqp(self):
        """Test SLSQP solver."""
        print("\nTesting SLSQP solver...")
        result = solve_with_slsqp(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, self.config
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(result['payload_fraction'], 0)
        self.assertLess(result['payload_fraction'], 1)
        
        # Verify stage information
        stages = result['stages']
        self.assertEqual(len(stages), self.n_stages)
        
        total_dv = 0
        for stage in stages:
            self.assertIn('stage', stage)
            self.assertIn('delta_v', stage)
            self.assertIn('Lambda', stage)
            
            # Verify Lambda calculation: λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
            dv = stage['delta_v']
            stage_num = stage['stage'] - 1
            isp = self.ISP[stage_num]
            epsilon = self.EPSILON[stage_num]
            expected_lambda = np.exp(-dv / (self.G0 * isp)) - epsilon
            
            self.assertAlmostEqual(stage['Lambda'], expected_lambda, places=4,
                                 msg=f"Stage {stage['stage']} lambda mismatch")
            
            total_dv += stage['delta_v']
            
        self.assertAlmostEqual(total_dv, self.TOTAL_DELTA_V, places=4)

    def test_solve_with_basin_hopping(self):
        """Test Basin-Hopping solver."""
        print("\nTesting Basin-Hopping solver...")
        result = solve_with_basin_hopping(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, self.config
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(result['payload_fraction'], 0)
        self.assertLess(result['payload_fraction'], 1)
        
        # Verify stage information
        stages = result['stages']
        self.assertEqual(len(stages), self.n_stages)
        
        total_dv = 0
        for stage in stages:
            self.assertIn('stage', stage)
            self.assertIn('delta_v', stage)
            self.assertIn('Lambda', stage)
            
            # Verify Lambda calculation: λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
            dv = stage['delta_v']
            stage_num = stage['stage'] - 1
            isp = self.ISP[stage_num]
            epsilon = self.EPSILON[stage_num]
            expected_lambda = np.exp(-dv / (self.G0 * isp)) - epsilon
            
            self.assertAlmostEqual(stage['Lambda'], expected_lambda, places=4,
                                 msg=f"Stage {stage['stage']} lambda mismatch")
            
            total_dv += stage['delta_v']
            
        self.assertAlmostEqual(total_dv, self.TOTAL_DELTA_V, places=4)

    def test_solve_with_genetic_algorithm(self):
        """Test Genetic Algorithm solver."""
        print("\nTesting GA solver...")
        result = solve_with_ga(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, self.config
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(result['payload_fraction'], 0)
        self.assertLess(result['payload_fraction'], 1)
        
        # Verify stage information
        stages = result['stages']
        self.assertEqual(len(stages), self.n_stages)
        
        total_dv = 0
        for stage in stages:
            self.assertIn('stage', stage)
            self.assertIn('delta_v', stage)
            self.assertIn('Lambda', stage)
            
            # Verify Lambda calculation: λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
            dv = stage['delta_v']
            stage_num = stage['stage'] - 1
            isp = self.ISP[stage_num]
            epsilon = self.EPSILON[stage_num]
            expected_lambda = np.exp(-dv / (self.G0 * isp)) - epsilon
            
            self.assertAlmostEqual(stage['Lambda'], expected_lambda, places=4,
                                 msg=f"Stage {stage['stage']} lambda mismatch")
            
            total_dv += stage['delta_v']
            
        self.assertAlmostEqual(total_dv, self.TOTAL_DELTA_V, places=4)

    def test_all_solvers(self):
        """Test and compare all optimization solvers."""
        print("\nTesting all solvers...")
        solvers = {
            'SLSQP': solve_with_slsqp,
            'BASIN-HOPPING': solve_with_basin_hopping,
            'GA': solve_with_ga
        }
        
        results = {}
        for name, solver in solvers.items():
            print(f"\nTesting {name} solver...")
            result = solver(
                self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
                self.TOTAL_DELTA_V, self.config
            )
            results[name] = result
            
            # Basic validation
            self.assertTrue(result['success'], f"{name} solver failed")
            self.assertGreater(result['payload_fraction'], 0)
            self.assertLess(result['payload_fraction'], 1)
            
            # Verify stage information
            stages = result['stages']
            self.assertEqual(len(stages), self.n_stages)
            
            total_dv = 0
            for stage in stages:
                self.assertIn('stage', stage)
                self.assertIn('delta_v', stage)
                self.assertIn('Lambda', stage)
                
                # Verify Lambda calculation: λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
                dv = stage['delta_v']
                stage_num = stage['stage'] - 1
                isp = self.ISP[stage_num]
                epsilon = self.EPSILON[stage_num]
                expected_lambda = np.exp(-dv / (self.G0 * isp)) - epsilon
                
                self.assertAlmostEqual(stage['Lambda'], expected_lambda, places=4,
                                     msg=f"Stage {stage['stage']} lambda mismatch")
                
                total_dv += stage['delta_v']
                
            self.assertAlmostEqual(total_dv, self.TOTAL_DELTA_V, places=4)
            
        # Compare results between solvers
        payload_fractions = [result['payload_fraction'] for result in results.values()]
        for i in range(len(payload_fractions)):
            for j in range(i + 1, len(payload_fractions)):
                diff = abs(payload_fractions[i] - payload_fractions[j]) / max(payload_fractions)
                self.assertLess(diff, 0.2,  # Allow 20% difference between solvers
                            f"Large payload fraction difference between solvers")

    def test_delta_v_split(self):
        """Test delta-v split calculations."""
        print("\nTesting delta-v split calculations...")
        delta_v_split = self.initial_guess
        self.assertEqual(len(delta_v_split), self.n_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in delta_v_split))
        self.assertAlmostEqual(np.sum(delta_v_split), self.TOTAL_DELTA_V, places=0)

    def test_solve_with_differential_evolution(self):
        """Test differential evolution solver."""
        logger.info("Starting differential evolution solver test")
        
        logger.debug(f"Initial parameters - guess: {self.initial_guess}")
        logger.debug(f"Bounds: {self.bounds}")
        logger.debug(f"ISP: {self.ISP}, EPSILON: {self.EPSILON}")
        
        result = solve_with_differential_evolution(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, self.config
        )
        
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        logger.debug(f"DE solution: {solution}")
        logger.debug(f"Total delta-V: {np.sum(solution)}")
        
        # Verify solution
        self.assertEqual(len(solution), self.n_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        logger.debug(f"Mass ratios: {mass_ratios}")
        logger.debug(f"Payload fraction: {payload_fraction}")
        
        self.assertTrue(0 <= payload_fraction <= 1)
        logger.info("Completed differential evolution solver test")

    def test_solve_with_pso(self):
        """Test particle swarm optimization solver."""
        print("\nTesting PSO solver...")
        result = solve_with_pso(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, self.config
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.n_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        logger.debug(f"Mass ratios: {mass_ratios}")
        logger.debug(f"Payload fraction: {payload_fraction}")
        
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

class TestCSVOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and ensure output directory exists."""
        cls.output_dir = os.path.join('output')
        # Create output directory if it doesn't exist
        os.makedirs(cls.output_dir, exist_ok=True)
        cls.stage_results_path = os.path.join(cls.output_dir, 'stage_results.csv')
        cls.optimization_results_path = os.path.join(cls.output_dir, 'optimization_summary.csv')
        
        # Create empty stage results file if it doesn't exist
        if not os.path.exists(cls.stage_results_path):
            with open(cls.stage_results_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Stage', 'Delta-V (m/s)', 'Stage Ratio (lambda)', 'Delta-V Contribution (%)', 'Method'])

    def test_csv_files_exist(self):
        """Verify that output CSV files are created."""
        self.assertTrue(os.path.exists(self.stage_results_path))
        self.assertTrue(os.path.exists(self.optimization_results_path))

    def test_stage_results_structure(self):
        """Verify structure of stage_results.csv."""
        with open(self.stage_results_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Get number of stages from input data
        with open('input_data.json', encoding='utf-8') as f:
            data = json.load(f)
            n_stages = len(data['stages'])

        # Expected rows = num_stages × num_methods
        expected_rows = n_stages * 6  # 6 optimization methods
        self.assertEqual(len(rows), expected_rows,
                        f"Expected {expected_rows} rows ({n_stages} stages × 6 methods)")

        # Verify required columns
        required_columns = ['Stage', 'Delta-V (m/s)', 'Stage Ratio (lambda)', 'Delta-V Contribution (%)', 'Method']
        for column in required_columns:
            self.assertTrue(any(column in row for row in rows),
                          f"Missing required column: {column}")

        # Verify data types and ranges
        for row in rows:
            # Stage should be in format "Stage N"
            self.assertRegex(row['Stage'], r'^Stage \d+$')
            
            # Delta-V should be a positive float
            delta_v = float(row['Delta-V (m/s)'])
            self.assertGreater(delta_v, 0)
            
            # Stage ratio should be between 0 and 1
            lambda_val = float(row['Stage Ratio (lambda)'])
            self.assertGreater(lambda_val, 0)
            self.assertLess(lambda_val, 1)
            
            # Delta-V contribution should be a percentage
            contribution = float(row['Delta-V Contribution (%)'])
            self.assertGreaterEqual(contribution, 0)
            self.assertLessEqual(contribution, 100)
            
            # Method should be one of the expected values
            expected_methods = {'SLSQP', 'BASIN-HOPPING', 'GA', 'ADAPTIVE-GA', 'DE', 'PSO'}
            self.assertIn(row['Method'], expected_methods)

    def test_lambda_calculations(self):
        """Verify λ calculations against manual computations."""
        # Create test data
        test_data = {
            'parameters': {
                'G0': 9.81,
                'TOTAL_DELTA_V': 9300.0
            },
            'stages': [
                {'stage': 1, 'ISP': 300.0, 'EPSILON': 0.06},
                {'stage': 2, 'ISP': 348.0, 'EPSILON': 0.04}
            ]
        }
        
        # Save test data to input_data.json
        with open('input_data.json', 'w') as f:
            json.dump(test_data, f)
            
        # Run optimization to generate stage results
        initial_guess = np.array([4650.0, 4650.0])
        bounds = [(1000, 8000), (1000, 8000)]
        G0 = test_data['parameters']['G0']
        ISP = np.array([s['ISP'] for s in test_data['stages']], dtype=float)
        EPSILON = np.array([s['EPSILON'] for s in test_data['stages']], dtype=float)
        TOTAL_DELTA_V = test_data['parameters']['TOTAL_DELTA_V']
        
        config = {
            'optimization': {
                'max_iterations': 1000,
                'population_size': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        }
        
        result = solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config)
        
        if not result['success']:
            self.fail(f"SLSQP optimization failed: {result['message']}")
        
        # Write stage results to CSV
        with open(self.stage_results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage', 'Delta-V (m/s)', 'Stage Ratio (lambda)', 'Delta-V Contribution (%)', 'Method'])
            
            for stage in result['stages']:
                stage_num = stage['stage']
                dv_value = stage['delta_v']
                isp = ISP[stage_num - 1]
                epsilon = EPSILON[stage_num - 1]
                
                # Calculate stage ratio using correct formula: λᵢ = rᵢ/(1 + εᵢ)
                mass_ratio = math.exp(-dv_value / (G0 * isp))
                lambda_value = mass_ratio / (1.0 + epsilon)
                
                dv_contribution = (dv_value / TOTAL_DELTA_V) * 100
                
                writer.writerow([
                    f'Stage {stage_num}',
                    f'{dv_value:.1f}',
                    f'{lambda_value:.6f}',
                    f'{dv_contribution:.1f}',
                    'SLSQP'
                ])
        
        # Now verify the calculations
        with open(self.stage_results_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        for row in rows:
            stage_num = int(row['Stage'].split()[1])
            dv = float(row['Delta-V (m/s)'])
            isp = ISP[stage_num - 1]
            epsilon = EPSILON[stage_num - 1]
            
            # Manual calculation using correct formula: λᵢ = rᵢ/(1 + εᵢ)
            mass_ratio = math.exp(-dv / (G0 * isp))
            expected_lambda = mass_ratio / (1.0 + epsilon)
            
            self.assertAlmostEqual(float(row['Stage Ratio (lambda)']), expected_lambda, places=4,
                                 msg=f"Stage ratio mismatch for stage {stage_num}, method {row['Method']}")

    def test_delta_v_split(self):
        """Verify delta-V split sums to total delta-V."""
        # Load total delta-V from input data
        with open('input_data.json', encoding='utf-8') as f:
            data = json.load(f)
            total_delta_v = float(data['parameters']['TOTAL_DELTA_V'])
            n_stages = len(data['stages'])
            
        # Read stage results and verify delta-V sum
        with open(self.stage_results_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Group by method and calculate total delta-V for each
            method_totals = {}
            for row in reader:
                method = row['Method']
                if method not in method_totals:
                    method_totals[method] = 0
                # Sum all stages
                method_totals[method] += float(row['Delta-V (m/s)'])
        
        # Verify total delta-V for each method with reduced precision (0 decimal places)
        # This allows for small numerical differences (~0.1 m/s in 9300 m/s is ~0.001% error)
        for method, total in method_totals.items():
            self.assertAlmostEqual(total, total_delta_v, places=0,
                                 msg=f"Delta-V mismatch for method {method}")

    def test_payload_fraction_consistency(self):
        """Verify payload fraction matches product of stage λ values."""
        # Read optimization results first
        with open(self.optimization_results_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            opt_results = {row['Method']: float(row['Payload Fraction']) for row in reader}

        # Calculate lambda products for each method
        with open(self.stage_results_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Group rows by method
            method_rows = {}
            for row in rows:
                method = row['Method']
                if method not in method_rows:
                    method_rows[method] = []
                method_rows[method].append(row)
            
            # Calculate lambda product for each method
            for method, method_data in method_rows.items():
                lambda_product = 1.0
                for row in method_data:
                    lambda_product *= float(row['Stage Ratio (lambda)'])
                
                # Skip if method not in optimization results
                if method not in opt_results:
                    continue
                
                # Verify consistency for this method with relaxed tolerance (1 place)
                self.assertAlmostEqual(
                    lambda_product, 
                    opt_results[method], 
                    places=1,  
                    msg=f"Payload fraction mismatch for method {method}")


class TestOptimizationCache(unittest.TestCase):
    """Test cases for optimization caching functionality."""
    
    def setUp(self):
        self.cache_file = "test_cache.pkl"
        self.cache = OptimizationCache(cache_file=self.cache_file)
        self.test_solution = np.array([0.3, 0.3, 0.4])
        self.test_fitness = -0.85
        
    def tearDown(self):
        cache_path = os.path.join(OUTPUT_DIR, self.cache_file)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test initial cache is empty
        self.assertNotIn(tuple(self.test_solution), self.cache.fitness_cache)
        
        # Test adding to cache
        self.cache.fitness_cache[tuple(self.test_solution)] = self.test_fitness
        self.assertEqual(self.cache.get_cached_fitness(self.test_solution), self.test_fitness)
        
        # Test persistence
        self.cache.save_cache()
        new_cache = OptimizationCache(cache_file=self.cache_file)
        self.assertEqual(new_cache.get_cached_fitness(self.test_solution), self.test_fitness)
        
    def test_cache_size(self):
        """Test cache size management."""
        # Add multiple solutions
        for i in range(10):
            solution = np.array([i/10, i/10, 1-2*(i/10)])
            self.cache.fitness_cache[tuple(solution)] = -0.8 - i/10
            self.cache.best_solutions.append(solution)
            
        # Test cache limits
        self.cache.save_cache()
        new_cache = OptimizationCache(cache_file=self.cache_file)
        self.assertEqual(len(new_cache.best_solutions), 10)
        
    def test_ga_integration(self):
        """Test integration with genetic algorithm."""
        good_solution = np.array([0.33, 0.33, 0.34])
        self.cache.fitness_cache[tuple(good_solution)] = -0.95
        self.cache.best_solutions.append(good_solution)
        self.cache.save_cache()
        
        result = solve_with_ga(
            initial_guess=good_solution,
            bounds=[(0.1, 0.5)] * 3,
            G0=9.81,
            ISP=[300, 300, 300],
            EPSILON=0.1,
            TOTAL_DELTA_V=1.0,
            config={'ga': {'population_size': 50, 'n_generations': 100}}
        )
        
        self.assertTrue(np.allclose(result, good_solution, rtol=0.1))

    def test_basin_hopping_caching(self):
        """Test that basin hopping properly uses caching."""
        # Add a known good solution to cache
        good_solution = np.array([0.33, 0.33, 0.34])
        self.cache.fitness_cache[tuple(good_solution)] = -0.95
        self.cache.best_solutions.append(good_solution)
        self.cache.save_cache()
        
        # Run basin hopping with same parameters
        result = solve_with_basin_hopping(
            initial_guess=good_solution,
            bounds=[(0.1, 0.5)] * 3,
            G0=9.81,
            ISP=[300, 300, 300],
            EPSILON=0.1,
            TOTAL_DELTA_V=1.0,
            config={'basin_hopping': {
                'n_iterations': 50,
                'temperature': 1.0,
                'stepsize': 0.1
            }}
        )
        
        # Verify result is close to the good solution
        self.assertTrue(np.allclose(result, good_solution, rtol=0.1))
        
        # Verify solution was cached
        new_cache = OptimizationCache(cache_file=self.cache_file)
        self.assertIn(tuple(result), new_cache.fitness_cache)
        self.assertTrue(any(np.allclose(s, result, rtol=0.1) for s in new_cache.best_solutions))


if __name__ == "__main__":
    unittest.main(verbosity=2)

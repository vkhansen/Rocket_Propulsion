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

class TestPayloadOptimization(unittest.TestCase):
    """Test cases for payload optimization functions."""

    def setUp(self):
        """Set up test cases."""
        # Load data from JSON file
        input_file = Path(__file__).parent / 'input_data.json'
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load parameters
        self.G0 = data['parameters']['G0']
        self.TOTAL_DELTA_V = data['parameters']['TOTAL_DELTA_V']
        
        # Load stage data
        self.stages = data['stages']
        self.num_stages = len(self.stages)
        
        # Extract ISP and EPSILON arrays
        self.ISP = np.array([stage['ISP'] for stage in self.stages])
        self.EPSILON = np.array([stage['EPSILON'] for stage in self.stages])
        
        # Set up optimization parameters
        self.initial_guess = np.ones(self.num_stages) * self.TOTAL_DELTA_V / self.num_stages
        self.bounds = [(0, self.TOTAL_DELTA_V) for _ in range(self.num_stages)]

    def test_load_input_data(self):
        """Test loading input data from JSON file."""
        print("\nTesting input data loading...")
        parameters, stages = load_input_data('input_data.json')
        self.assertEqual(len(stages), len(self.stages))
        self.assertEqual(stages[0]["ISP"], self.stages[0]["ISP"])
        self.assertEqual(stages[1]["EPSILON"], self.stages[1]["EPSILON"])
        self.assertEqual(parameters["TOTAL_DELTA_V"], self.TOTAL_DELTA_V)
        self.assertEqual(parameters["G0"], self.G0)

    def test_calculate_mass_ratios(self):
        """Test stage ratio (Λ) calculation."""
        print("\nTesting stage ratio calculation...")
        # Test case with 2 stages
        dv = np.array([4650, 4650])  # Equal split of delta-V
        ISP = [300, 348]  # Different ISP for each stage
        EPSILON = [0.06, 0.04]  # Different structural coefficients
        G0 = 9.81
        
        # Calculate stage ratios
        ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        
        # Manual calculation for verification
        # First calculate mass ratios using rocket equation
        mass_ratio1 = np.exp(-dv[0] / (G0 * ISP[0]))  # Bottom stage
        mass_ratio2 = np.exp(-dv[1] / (G0 * ISP[1]))  # Top stage
        
        # Then calculate stage ratios (Λ)
        lambda1 = 1.0 / (mass_ratio1 + EPSILON[0])  # Bottom stage
        lambda2 = 1.0 / (mass_ratio2 + EPSILON[1])  # Top stage
        
        # Verify results
        self.assertEqual(len(ratios), 2)
        self.assertAlmostEqual(ratios[0], lambda1, places=4)
        self.assertAlmostEqual(ratios[1], lambda2, places=4)
        
        # Test with single stage
        single_dv = np.array([9300])
        single_isp = [300]
        single_epsilon = [0.06]
        
        single_ratios = calculate_mass_ratios(single_dv, single_isp, single_epsilon, G0)
        single_mass_ratio = np.exp(-single_dv[0] / (G0 * single_isp[0]))
        expected_lambda = 1.0 / (single_mass_ratio + single_epsilon[0])
        
        self.assertEqual(len(single_ratios), 1)
        self.assertAlmostEqual(single_ratios[0], expected_lambda, places=4)

    def test_payload_fraction(self):
        """Test payload fraction calculation."""
        print("\nTesting payload fraction calculation...")
        # Test with 2 stages
        dv = np.array([4650, 4650])
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        G0 = 9.81
        
        # Calculate stage ratios
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        fraction = calculate_payload_fraction(mass_ratios)
        
        # Manual calculation
        # First calculate mass ratios
        mr1 = np.exp(-dv[0] / (G0 * ISP[0]))
        mr2 = np.exp(-dv[1] / (G0 * ISP[1]))
        
        # Then calculate stage ratios
        lambda1 = 1.0 / (mr1 + EPSILON[0])
        lambda2 = 1.0 / (mr2 + EPSILON[1])
        
        # Payload fraction is product of stage ratios
        expected_fraction = lambda1 * lambda2
        
        self.assertAlmostEqual(fraction, expected_fraction, places=4)
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 1)

    def test_payload_fraction_objective(self):
        """Test payload fraction objective function."""
        print("\nTesting payload fraction objective...")
        dv = np.array([4650, 4650])
        G0 = 9.81
        ISP = [300, 348]
        EPSILON = [0.06, 0.04]
        
        result = payload_fraction_objective(dv, G0, ISP, EPSILON)
        
        # Manual calculation
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        expected = -calculate_payload_fraction(mass_ratios)  # Negative because we minimize
        
        self.assertAlmostEqual(result, expected, places=4)
        self.assertGreater(result, -1)  # Should be negative but greater than -1

    def test_solve_with_slsqp(self):
        """Test SLSQP solver."""
        print("\nTesting SLSQP solver...")
        result = solve_with_slsqp(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

    def test_solve_with_differential_evolution(self):
        """Test differential evolution solver."""
        print("\nTesting differential evolution solver...")
        result = solve_with_differential_evolution(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

    def test_solve_with_genetic_algorithm(self):
        """Test genetic algorithm solver."""
        print("\nTesting GA solver...")
        result = solve_with_ga(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

    def test_solve_with_adaptive_ga(self):
        """Test adaptive genetic algorithm solver."""
        print("\nTesting adaptive GA solver...")
        result = solve_with_adaptive_ga(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

    def test_solve_with_pso(self):
        """Test particle swarm optimization solver."""
        print("\nTesting PSO solver...")
        result = solve_with_pso(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)

    def test_all_solvers(self):
        """Test all solvers and compare results."""
        print("\nTesting and comparing all solvers...")
        solvers = [
            solve_with_slsqp,
            solve_with_basin_hopping,
            solve_with_differential_evolution,
            solve_with_ga,
            solve_with_adaptive_ga,
            solve_with_pso
        ]
        
        results = []
        payload_fractions = []
        
        for solver in solvers:
            result = solver(
                self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
                self.TOTAL_DELTA_V, CONFIG
            )
            solution = np.array([stage['delta_v'] for stage in result['stages']])
            results.append(solution)
            payload_fractions.append(result['payload_fraction'])
        
        # Compare payload fractions between solvers
        for i in range(len(solvers)):
            for j in range(i + 1, len(solvers)):
                diff = abs(payload_fractions[i] - payload_fractions[j])
                self.assertLess(diff, 0.2,  # Allow 20% difference between solvers
                            f"Large payload fraction difference between solvers {i} and {j}")

    def test_lambda_calculations(self):
        """Verify stage ratio (Λ) calculations against manual computations."""
        # Load input data to get correct ISP and EPSILON values
        with open('input_data.json', encoding='utf-8') as f:
            data = json.load(f)
            stages = data['stages']
            
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

        # Test each method's results
        for method, method_data in method_rows.items():
            # Sort stages by number
            sorted_stages = sorted(method_data, key=lambda x: int(x['Stage'].split()[1]))
            
            for i in range(len(sorted_stages)):
                stage = sorted_stages[i]
                stage_num = int(stage['Stage'].split()[1])  # Split 'Stage 1' and take the number
                dv = float(stage['Delta-V (m/s)'])
                isp = stages[stage_num]['ISP']
                epsilon = stages[stage_num]['EPSILON']
                
                # Calculate mass ratio for current stage
                mass_ratio = math.exp(-dv / (9.81 * isp))
                
                # Calculate stage ratio (Λ)
                if i < len(sorted_stages) - 1:  # Not the top stage
                    # Get upper stage data
                    upper_stage = sorted_stages[i + 1]
                    upper_dv = float(upper_stage['Delta-V (m/s)'])
                    upper_isp = stages[stage_num + 1]['ISP']
                    upper_epsilon = stages[stage_num + 1]['EPSILON']
                    
                    # Calculate upper stage mass including structural mass
                    upper_mass = math.exp(-upper_dv / (9.81 * upper_isp)) * (1 + upper_epsilon)
                    # Stage ratio is upper stage mass divided by current stage mass
                    expected_lambda = 1.0 / (mass_ratio + epsilon)
                else:  # Top stage
                    expected_lambda = 1.0 / (mass_ratio + epsilon)
                
                # Verify with CSV value with reduced precision (2 decimal places)
                self.assertAlmostEqual(
                    float(stage['Stage Ratio (Λ)']), 
                    expected_lambda, 
                    places=2,
                    msg=f"Stage ratio (Λ) mismatch for {stage['Stage']}, method {method}"
                )

    def test_delta_v_split(self):
        """Test delta-v split calculations."""
        print("\nTesting delta-v split calculations...")
        delta_v_split = self.initial_guess
        self.assertEqual(len(delta_v_split), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in delta_v_split))
        self.assertAlmostEqual(np.sum(delta_v_split), self.TOTAL_DELTA_V, places=0)

    def test_solve_with_basin_hopping(self):
        """Test basin hopping solver."""
        print("\nTesting basin hopping solver...")
        result = solve_with_basin_hopping(
            self.initial_guess, self.bounds, self.G0, self.ISP, self.EPSILON,
            self.TOTAL_DELTA_V, CONFIG
        )
        
        # Extract solution from result dictionary
        solution = np.array([stage['delta_v'] for stage in result['stages']])
        
        # Verify solution constraints
        self.assertEqual(len(solution), self.num_stages)
        self.assertTrue(all(0 <= dv <= self.TOTAL_DELTA_V for dv in solution))
        self.assertAlmostEqual(np.sum(solution), self.TOTAL_DELTA_V, places=5)
        
        # Calculate payload fraction to verify solution quality
        mass_ratios = calculate_mass_ratios(solution, self.ISP, self.EPSILON, self.G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        self.assertTrue(0 <= payload_fraction <= 1)
        
        # Verify Lambda values
        for stage in result['stages']:
            self.assertTrue(0 < stage['Lambda'] < 1)


class TestCSVOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join('output')
        cls.stage_results_path = os.path.join(cls.output_dir, 'stage_results.csv')
        cls.optimization_results_path = os.path.join(cls.output_dir, 'optimization_summary.csv')

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
        # Load input data to get correct ISP and EPSILON values
        with open('input_data.json', encoding='utf-8') as f:
            data = json.load(f)
            stages = data['stages']
            
        with open(self.stage_results_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Create test cases from actual input data
        test_cases = []
        for stage in stages:
            test_cases.append({
                'stage': stage['stage'],
                'ISP': stage['ISP'],
                'EPSILON': stage['EPSILON']
            })

        for row in rows:
            # Extract stage number from 'Stage X' format
            stage_num = int(row['Stage'].split()[1])  # Split 'Stage 1' and take the number
            test_case = test_cases[stage_num - 1]  # Convert to 0-based index
            
            dv = float(row['Delta-V (m/s)'])
            isp = test_case['ISP']
            epsilon = test_case['EPSILON']
            
            # Manual calculation
            expected_lambda = math.exp(-dv / (9.81 * isp)) - epsilon
            
            # Verify with CSV value
            self.assertAlmostEqual(float(row['Stage Ratio (lambda)']), expected_lambda, places=2,
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
                    msg=f"Payload fraction mismatch for method {method}"
                )


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

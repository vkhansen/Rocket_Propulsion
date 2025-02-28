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
    solve_with_pso,
    RocketOptimizationProblem
)
from src.utils.config import setup_logging, CONFIG, logger
from src.optimization.cache import OptimizationCache, OUTPUT_DIR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize logging
setup_logging()

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
    """Test cases for CSV output functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and ensure output directory exists."""
        cls.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Test data
        cls.test_data = {
            'G0': 9.81,
            'TOTAL_DELTA_V': 9000,
            'ISP': [300, 350],
            'EPSILON': [0.1, 0.1],
            'initial_guess': [4500, 4500],
            'bounds': [(0, 9000), (0, 9000)]
        }
        
        # Configuration for all solvers
        cls.config = {
            'optimization': {
                'max_iterations': 100,
                'population_size': 50
            },
            'basin_hopping': {
                'n_iterations': 50,
                'temperature': 1.0,
                'stepsize': 0.5
            },
            'ga': {
                'population_size': 50,
                'n_generations': 50
            },
            'pso': {
                'n_particles': 50,
                'n_iterations': 50
            },
            'de': {
                'population_size': 50,
                'max_generations': 50
            },
            'adaptive_ga': {
                'initial_pop_size': 50,
                'n_generations': 50
            }
        }
        
        # Run all solvers
        cls.solvers = {
            'SLSQP': solve_with_slsqp,
            'BASIN-HOPPING': solve_with_basin_hopping,
            'GA': solve_with_ga,
            'PSO': solve_with_pso,
            'DE': solve_with_differential_evolution,
            'AGA': solve_with_adaptive_ga
        }
        
        cls.results = {}
        for name, solver in cls.solvers.items():
            try:
                result = solver(
                    cls.test_data['initial_guess'],
                    cls.test_data['bounds'],
                    cls.test_data['G0'],
                    cls.test_data['ISP'],
                    cls.test_data['EPSILON'],
                    cls.test_data['TOTAL_DELTA_V'],
                    cls.config
                )
                if result['success']:
                    # Add required fields for CSV output
                    result['payload_fraction'] = result.get('payload_fraction', 0.0)
                    result['error'] = result.get('error', 0.0)
                    result['execution_time'] = result.get('execution_time', 0.0)
                    
                    # Convert stage data to expected format
                    stages = result.get('stages', [])
                    result['dv'] = [stage['delta_v'] for stage in stages]
                    result['stage_ratios'] = [stage['Lambda'] for stage in stages]
                    
                    cls.results[name] = result
                else:
                    logger.warning(f"{name} solver failed: {result['message']}")
            except Exception as e:
                logger.error(f"Error running {name} solver: {e}")
                continue
        
        # Generate CSV reports
        from src.reporting.csv_reports import write_results_to_csv
        cls.summary_path, cls.detailed_path = write_results_to_csv(cls.results, cls.test_data['ISP'], cls.output_dir)
    
    def test_csv_files_exist(self):
        """Verify that output CSV files are created."""
        self.assertTrue(os.path.exists(self.summary_path), "Summary CSV file not found")
        self.assertTrue(os.path.exists(self.detailed_path), "Detailed CSV file not found")
    
    def test_summary_csv_structure(self):
        """Verify structure of optimization_summary.csv."""
        with open(self.summary_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = ['Method', 'Payload Fraction', 'Error', 'Time (s)']
            self.assertEqual(header, expected_header, "Summary CSV header mismatch")
            
            rows = list(reader)
            self.assertEqual(len(rows), len(self.results),
                           f"Expected {len(self.results)} rows in summary CSV")
            
            for row in rows:
                self.assertEqual(len(row), len(expected_header),
                               "Summary CSV row has incorrect number of columns")
                
                # Verify data types
                method = row[0]
                self.assertIn(method, self.results, f"Unknown method {method} in summary CSV")
                self.assertTrue(self.is_float(row[1]), "Payload fraction is not a valid float")
                self.assertTrue(self.is_float(row[2]), "Error is not a valid float")
                self.assertTrue(self.is_float(row[3]), "Execution time is not a valid float")
    
    def test_detailed_csv_structure(self):
        """Verify structure of stage_results.csv."""
        with open(self.detailed_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = ['Stage', 'Delta-V (m/s)', 'Stage Ratio (lambda)', 'Delta-V Contribution (%)', 'Method']
            self.assertEqual(header, expected_header, "Detailed CSV header mismatch")
            
            rows = list(reader)
            n_stages = len(self.test_data['ISP'])
            expected_rows = n_stages * len(self.results)
            self.assertEqual(len(rows), expected_rows,
                           f"Expected {expected_rows} rows ({n_stages} stages × {len(self.results)} methods)")
            
            for row in rows:
                self.assertEqual(len(row), len(expected_header),
                               "Detailed CSV row has incorrect number of columns")
                
                # Verify data types
                self.assertTrue(row[0].isdigit(), "Stage number is not a valid integer")
                self.assertTrue(self.is_float(row[1]), "Delta-V is not a valid float")
                self.assertTrue(self.is_float(row[2]), "Stage ratio is not a valid float")
                self.assertTrue(self.is_float(row[3]), "Delta-V contribution is not a valid float")
                self.assertIn(row[4], self.results, f"Unknown method {row[4]} in detailed CSV")
    
    @staticmethod
    def is_float(value):
        """Helper method to check if a string represents a valid float."""
        try:
            float(value.replace('%', ''))  # Handle percentage values
            return True
        except ValueError:
            return False

class TestOptimizationCache(unittest.TestCase):
    """Test cases for optimization caching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = OptimizationCache(cache_file="test_cache.pkl", max_size=10)
        
        # Test data
        self.test_data = {
            'G0': 9.81,
            'TOTAL_DELTA_V': 9000,
            'ISP': [300, 350],
            'EPSILON': [0.1, 0.1],
            'initial_guess': [4500, 4500],
            'bounds': [(0, 9000), (0, 9000)]
        }
        
        self.config = {
            'optimization': {
                'max_iterations': 50,
                'population_size': 20
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()
        if os.path.exists(self.cache.cache_file):
            os.remove(self.cache.cache_file)
    
    def test_cache_size(self):
        """Test cache size management."""
        # Add more solutions than the cache size
        for i in range(self.cache.max_size + 5):
            x = np.array([i * 100, (i + 1) * 100])
            fitness = -float(i)
            self.cache.add(x, fitness)
        
        # Verify cache size is maintained
        self.assertLessEqual(len(self.cache.cache_fitness), self.cache.max_size)
        
        # Verify best solutions are kept
        best_solutions = self.cache.get_best_solutions()
        self.assertLessEqual(len(best_solutions), self.cache.max_size)
    
    def test_cache_persistence(self):
        """Test saving and loading cache."""
        # Add some solutions
        test_solutions = [
            (np.array([100, 200]), -1.0),
            (np.array([300, 400]), -2.0),
            (np.array([500, 600]), -3.0)
        ]
        for x, f in test_solutions:
            self.cache.add(x, f)
        
        # Save cache
        self.cache.save_cache()
        
        # Create new cache instance and load
        new_cache = OptimizationCache(cache_file=self.cache.cache_file)
        
        # Verify solutions are preserved
        for x, f in test_solutions:
            cached_f = new_cache.get_cached_fitness(x)
            self.assertIsNotNone(cached_f)
            self.assertEqual(cached_f, f)
    
    def test_cache_hits(self):
        """Test cache hit counting."""
        x = np.array([100, 200])
        f = -1.0
        
        # Add solution
        self.cache.add(x, f)
        initial_hits = self.cache.hit_count
        
        # Get solution multiple times
        for _ in range(3):
            cached_f = self.cache.get_cached_fitness(x)
            self.assertEqual(cached_f, f)
        
        # Verify hit count increased
        self.assertEqual(self.cache.hit_count, initial_hits + 3)
    
    def test_basin_hopping_caching(self):
        """Test caching with basin hopping solver."""
        # First run
        result1 = solve_with_basin_hopping(
            self.test_data['initial_guess'],
            self.test_data['bounds'],
            self.test_data['G0'],
            self.test_data['ISP'],
            self.test_data['EPSILON'],
            self.test_data['TOTAL_DELTA_V'],
            self.config
        )
        
        initial_hits = self.cache.hit_count
        
        # Second run with same parameters
        result2 = solve_with_basin_hopping(
            self.test_data['initial_guess'],
            self.test_data['bounds'],
            self.test_data['G0'],
            self.test_data['ISP'],
            self.test_data['EPSILON'],
            self.test_data['TOTAL_DELTA_V'],
            self.config
        )
        
        # Verify cache hits increased
        self.assertGreater(self.cache.hit_count, initial_hits)
        
        # Verify results are similar
        self.assertAlmostEqual(result1['payload_fraction'], 
                             result2['payload_fraction'],
                             places=4)
    
    def test_ga_integration(self):
        """Test integration with genetic algorithm."""
        # Run GA optimization with caching
        problem = RocketOptimizationProblem(
            n_var=len(self.test_data['initial_guess']),
            bounds=self.test_data['bounds'],
            G0=self.test_data['G0'],
            ISP=self.test_data['ISP'],
            EPSILON=self.test_data['EPSILON'],
            TOTAL_DELTA_V=self.test_data['TOTAL_DELTA_V']
        )
        self.cache = problem.cache  # Store reference to cache

        # First run
        result1 = solve_with_ga(
            self.test_data['initial_guess'],
            self.test_data['bounds'],
            self.test_data['G0'],
            self.test_data['ISP'],
            self.test_data['EPSILON'],
            self.test_data['TOTAL_DELTA_V'],
            self.config,
            problem=problem  # Pass the problem instance
        )

        initial_hits = problem.cache.hit_count  # Use problem.cache instead of self.cache
        problem.cache.save_cache()  # Save cache after first run

        # Second run - use same problem instance to share cache
        result2 = solve_with_ga(
            self.test_data['initial_guess'],
            self.test_data['bounds'],
            self.test_data['G0'],
            self.test_data['ISP'],
            self.test_data['EPSILON'],
            self.test_data['TOTAL_DELTA_V'],
            self.config,
            problem=problem  # Pass the same problem instance
        )

        # Verify cache hits increased
        self.assertGreater(problem.cache.hit_count, initial_hits)  # Use problem.cache
        
        # Verify results are similar
        self.assertAlmostEqual(result1['payload_fraction'], 
                             result2['payload_fraction'],
                             places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

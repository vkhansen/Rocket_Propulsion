import os
import tempfile
import unittest
import numpy as np
import logging

# Configure logging: output debug messages to both console and a file.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                # Console output
        logging.FileHandler("test_debug.log", mode='w')  # File output (overwrites each run)
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.INFO)



# Import the functions from your main module.
# Adjust the module name if needed.
import payload_optimization as po
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPayloadOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("Starting TestPayloadOptimization suite.")

    def setUp(self):
        logger.info("Setting up temporary CSV file for test.")
        # Create a temporary CSV file for testing read_csv_input.
        self.sample_csv_content = """Parameter,Value
G0,9.81
TOTAL_DELTA_V,9500

stage,ISP,EPSILON
1,300,0.1
2,320,0.08
"""
        self.temp_csv = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="")
        self.temp_csv.write(self.sample_csv_content)
        self.temp_csv.close()  # Flush and close so that it can be read by the function.
        logger.debug(f"Temporary CSV file created at: {self.temp_csv.name}")

    def tearDown(self):
        logger.info("Tearing down temporary CSV file.")
        # Remove the temporary CSV file.
        os.remove(self.temp_csv.name)
        logger.debug(f"Temporary CSV file removed: {self.temp_csv.name}")

    def test_read_csv_input(self):
        logger.info("Running test_read_csv_input")
        parameters, stages = po.read_csv_input(self.temp_csv.name)
        logger.debug(f"Parameters read: {parameters}")
        logger.debug(f"Stages read: {stages}")

        self.assertIn("G0", parameters)
        self.assertIn("TOTAL_DELTA_V", parameters)
        self.assertAlmostEqual(float(parameters["G0"]), 9.81, places=5)
        self.assertAlmostEqual(float(parameters["TOTAL_DELTA_V"]), 9500, places=5)

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["stage"], 1)
        self.assertAlmostEqual(stages[0]["ISP"], 300.0, places=5)
        self.assertAlmostEqual(stages[0]["EPSILON"], 0.1, places=5)

    def test_payload_fraction_objective_valid(self):
        logger.info("Running test_payload_fraction_objective_valid")
        # For a one-stage case.
        G0 = 9.81
        dv = [1000]  # m/s allocation
        ISP = [300]
        EPSILON = [0.1]
        f = po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        logger.debug(f"Computed objective: {f}")
        expected_ratio = np.exp(-1000 / (G0 * ISP[0])) - EPSILON[0]
        logger.debug(f"Expected stage ratio: {expected_ratio}")
        # The function returns the negative product (here just the negative of one value).
        self.assertAlmostEqual(f, -expected_ratio, places=5)

    def test_payload_fraction_objective_negative_isp(self):
        logger.info("Running test_payload_fraction_objective_negative_isp")
        G0 = 9.81
        dv = [1000]
        ISP = [-300]  # Negative ISP
        EPSILON = [0.1]
        
        with self.assertRaises(ValueError) as context:
            po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        self.assertTrue("ISP values must be positive" in str(context.exception))

    def test_payload_fraction_objective_overflow(self):
        logger.info("Running test_payload_fraction_objective_overflow")
        G0 = 9.81
        dv = [1e6]  # Very large delta-V
        ISP = [100]  # Small ISP to force overflow
        EPSILON = [0.1]
        penalty_coeff = 1e6
        
        result = po.payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff)
        self.assertEqual(result, penalty_coeff)

    def test_objective_with_penalty(self):
        logger.info("Running test_objective_with_penalty")
        G0 = 9.81
        dv = [1000, 2000]  # m/s allocations; sum is 3000 m/s.
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        total_delta_v = 3500  # Constraint violation: sum(dv) is 500 m/s too low.
        base_obj = po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        logger.debug(f"Base objective without penalty: {base_obj}")
        penalty_coeff = 1e6
        expected_penalty = penalty_coeff * 500  # Because error = 3500 - 3000 = 500.
        logger.debug(f"Expected penalty: {expected_penalty}")
        obj_with_penalty = po.objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff=penalty_coeff)
        logger.debug(f"Objective with penalty: {obj_with_penalty}")
        self.assertAlmostEqual(obj_with_penalty, base_obj + expected_penalty, places=5)

    def test_objective_with_penalty_nan_input(self):
        logger.info("Running test_objective_with_penalty_nan_input")
        G0 = 9.81
        dv = [np.nan, 1000]  # Include NaN value
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        total_delta_v = 3000
        penalty_coeff = 1e6
        
        result = po.objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff)
        self.assertEqual(result, penalty_coeff)

    def test_objective_with_penalty_scaled_tolerance(self):
        logger.info("Running test_objective_with_penalty_scaled_tolerance")
        G0 = 9.81
        dv = [1500, 1500.1]  # Sum is 3000.1
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        total_delta_v = 3000
        penalty_coeff = 1e6
        tol = 1.0  # Large tolerance
        
        # With large tolerance, small deviation should not incur penalty
        result_with_tol = po.objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, 
                                                  penalty_coeff, tol)
        # With default tolerance, same deviation should incur penalty
        result_default = po.objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, 
                                                 penalty_coeff)
        
        self.assertLess(result_with_tol, result_default)

    def test_optimize_payload_allocation_one_stage(self):
        logger.info("Running test_optimize_payload_allocation_one_stage")
        G0 = 9.81
        TOTAL_DELTA_V = 6500  # A physically achievable value for one stage
        ISP = [300]
        EPSILON = [0.1]

        optimal_dv, stage_ratios, overall_payload = po.optimize_payload_allocation(
            TOTAL_DELTA_V, ISP, EPSILON, G0, method="SLSQP"
        )
        logger.debug(f"Optimal DV allocations: {optimal_dv}")
        logger.debug(f"Stage ratios: {stage_ratios}")
        logger.debug(f"Overall payload fraction: {overall_payload}")
        # For one stage, the allocation should equal TOTAL_DELTA_V.
        self.assertAlmostEqual(optimal_dv[0], TOTAL_DELTA_V, places=5)
        expected_ratio = np.exp(-TOTAL_DELTA_V / (G0 * ISP[0])) - EPSILON[0]
        self.assertAlmostEqual(stage_ratios[0], expected_ratio, places=5)

    def test_optimize_payload_allocation_two_stage(self):
        logger.info("Running test_optimize_payload_allocation_two_stage")
        G0 = 9.81
        TOTAL_DELTA_V = 9500
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        methods = ["SLSQP", "differential_evolution", "GA"]

        for method in methods:
            with self.subTest(method=method):
                logger.info(f"Testing optimization with method: {method}")
                optimal_dv, stage_ratios, overall_payload = po.optimize_payload_allocation(
                    TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
                )
                logger.debug(f"Method: {method} optimal DV: {optimal_dv}")
                logger.debug(f"Method: {method} stage ratios: {stage_ratios}")
                logger.debug(f"Method: {method} overall payload: {overall_payload}")
                self.assertAlmostEqual(np.sum(optimal_dv), TOTAL_DELTA_V, places=5)
                for ratio in stage_ratios:
                    self.assertGreater(ratio, 0)
                self.assertGreater(overall_payload, 0)

    def test_optimize_payload_allocation_invalid_total_delta_v(self):
        logger.info("Running test_optimize_payload_allocation_invalid_total_delta_v")
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        # Compute maximum delta-V for each stage.
        max_dv_stage1 = -G0 * 300 * np.log(0.1)
        max_dv_stage2 = -G0 * 320 * np.log(0.08)
        max_total_dv = max_dv_stage1 + max_dv_stage2
        TOTAL_DELTA_V = max_total_dv + 1  # Exceeds the maximum possible.
        logger.debug(f"Computed maximum total DV: {max_total_dv}")
        with self.assertRaises(ValueError):
            po.optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0, method="SLSQP")

    def test_plot_dv_breakdown(self):
        logger.info("Running test_plot_dv_breakdown")
        # Create a dummy results list.
        results = [
                    {
                "Method": "TestMethod",
                "dv": [3000, 3500],
                "ratio": [0.5, 0.6],
                "Payload Fraction": 0.3,  # if needed by consistency elsewhere
                "Time (s)": 0.1,          # if needed by consistency elsewhere
            }
        ]

        plot_filename = "test_dv_breakdown.png"
        po.plot_dv_breakdown(results, total_delta_v=9500, gravity_loss=100, drag_loss=50, filename=plot_filename)
        logger.debug(f"Plot file created: {plot_filename}")
        self.assertTrue(os.path.exists(plot_filename))
        os.remove(plot_filename)
        logger.debug(f"Plot file removed: {plot_filename}")


if __name__ == "__main__":
    # Run tests in verbose mode.
    unittest.main(verbosity=2)

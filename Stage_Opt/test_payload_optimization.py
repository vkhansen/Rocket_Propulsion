import os
import tempfile
import unittest
import numpy as np

# Import the functions from your main module.
# Adjust the module name if needed.
import payload_optimization as po
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestPayloadOptimization(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        # Remove the temporary CSV file.
        os.remove(self.temp_csv.name)

    def test_read_csv_input(self):
        """Test that the CSV file is correctly parsed into parameters and stage data."""
        parameters, stages = po.read_csv_input(self.temp_csv.name)
        self.assertIn("G0", parameters)
        self.assertIn("TOTAL_DELTA_V", parameters)
        self.assertAlmostEqual(float(parameters["G0"]), 9.81, places=5)
        self.assertAlmostEqual(float(parameters["TOTAL_DELTA_V"]), 9500, places=5)

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["stage"], 1)
        self.assertAlmostEqual(stages[0]["ISP"], 300.0, places=5)
        self.assertAlmostEqual(stages[0]["EPSILON"], 0.1, places=5)

    def test_payload_fraction_objective_valid(self):
        """Test payload_fraction_objective for a valid one-stage input."""
        # For a one-stage case.
        G0 = 9.81
        dv = [1000]  # m/s allocation
        ISP = [300]
        EPSILON = [0.1]
        f = po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        # Compute the expected stage ratio:
        expected_ratio = np.exp(-1000 / (G0 * ISP[0])) - EPSILON[0]
        # The function returns the negative product (here just the negative of one value).
        self.assertAlmostEqual(f, -expected_ratio, places=5)

    def test_objective_with_penalty(self):
        """Test that objective_with_penalty adds the correct penalty when the equality constraint is violated."""
        G0 = 9.81
        dv = [1000, 2000]  # m/s allocations; sum is 3000 m/s.
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        total_delta_v = 3500  # Constraint violation: sum(dv) is 500 m/s too low.
        base_obj = po.payload_fraction_objective(dv, G0, ISP, EPSILON)
        penalty_coeff = 1e6
        expected_penalty = penalty_coeff * 500  # Because error = 3500 - 3000 = 500.
        obj_with_penalty = po.objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff=penalty_coeff)
        self.assertAlmostEqual(obj_with_penalty, base_obj + expected_penalty, places=5)

    def test_optimize_payload_allocation_one_stage(self):
        """Test optimization with a one-stage rocket; the entire ΔV should go to the single stage."""
        G0 = 9.81
        TOTAL_DELTA_V = 6500
        ISP = [300]
        EPSILON = [0.1]

        optimal_dv, stage_ratios, overall_payload = po.optimize_payload_allocation(
            TOTAL_DELTA_V, ISP, EPSILON, G0, method="SLSQP"
        )
        # For one stage, the allocation should equal TOTAL_DELTA_V.
        self.assertAlmostEqual(optimal_dv[0], TOTAL_DELTA_V, places=5)
        # Also check that the computed stage ratio matches the expected value.
        expected_ratio = np.exp(-TOTAL_DELTA_V / (G0 * ISP[0])) - EPSILON[0]
        self.assertAlmostEqual(stage_ratios[0], expected_ratio, places=5)

    def test_optimize_payload_allocation_two_stage(self):
        """Test optimization with a two-stage rocket using different methods."""
        G0 = 9.81
        TOTAL_DELTA_V = 9500
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        methods = ["SLSQP", "differential_evolution", "GA"]

        for method in methods:
            with self.subTest(method=method):
                optimal_dv, stage_ratios, overall_payload = po.optimize_payload_allocation(
                    TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
                )
                # Check that the sum of delta-V allocations equals TOTAL_DELTA_V.
                self.assertAlmostEqual(np.sum(optimal_dv), TOTAL_DELTA_V, places=5)
                # Check that each stage ratio is positive (i.e. a physically meaningful result).
                for ratio in stage_ratios:
                    self.assertGreater(ratio, 0)
                # Optionally, overall payload fraction should be positive.
                self.assertGreater(overall_payload, 0)

    def test_optimize_payload_allocation_invalid_total_delta_v(self):
        """Test that an error is raised if the requested TOTAL_DELTA_V is impossible."""
        G0 = 9.81
        ISP = [300, 320]
        EPSILON = [0.1, 0.08]
        # Compute maximum delta-V for each stage: -G0 * isp * log(epsilon)
        max_dv_stage1 = -G0 * 300 * np.log(0.1)
        max_dv_stage2 = -G0 * 320 * np.log(0.08)
        max_total_dv = max_dv_stage1 + max_dv_stage2
        # Choose a total_delta_v that is too high.
        TOTAL_DELTA_V = max_total_dv + 1
        with self.assertRaises(ValueError):
            po.optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0, method="SLSQP")

    def test_plot_dv_breakdown(self):
        """Test that the plotting function creates a file as expected."""
        # Create a dummy result list.
        results = [
            {
                "method": "TestMethod",
                "dv": [3000, 3500],
                "ratio": [0.5, 0.6],
                "payload": 0.3,
                "time": 0.1,
            }
        ]
        plot_filename = "test_dv_breakdown.png"
        # Call the plotting function.
        po.plot_dv_breakdown(results, total_delta_v=9500, gravity_loss=100, drag_loss=50, filename=plot_filename)
        # Check that the file was created.
        self.assertTrue(os.path.exists(plot_filename))
        # Clean up the generated file.
        os.remove(plot_filename)


if __name__ == "__main__":
    unittest.main()

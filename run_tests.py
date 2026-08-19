"""
LeafDoc Automated Verification & Test Suite Runner
Runs all unit and regression tests across data, models, training engine, reports, and inference.

Usage:
    python run_tests.py
"""
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    print("=" * 72)
    print("       Running LeafDoc Full System Verification & Test Suite")
    print("=" * 72)

    # Discover and run all test cases in the 'tests' directory
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).parent / "tests")
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 72)
    if result.wasSuccessful():
        print(f"  [SUCCESS] All {result.testsRun} tests PASSED successfully!")
    else:
        print(f"  [FAILURE] {len(result.failures)} failures, {len(result.errors)} errors encountered.")
    print("=" * 72)

    sys.exit(0 if result.wasSuccessful() else 1)

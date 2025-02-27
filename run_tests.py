#!/usr/bin/env python
"""
Run all tests for DualPipe.

This script discovers and runs all the tests in the test directory.
"""

import os
import sys
import unittest
import argparse


def run_tests(verbose=False, pattern=None, failfast=False):
    """
    Run all tests.
    
    Args:
        verbose: Whether to show verbose output.
        pattern: Pattern to match test files.
        failfast: Whether to stop at the first failure.
    """
    # Set up test discovery
    loader = unittest.TestLoader()
    
    if pattern:
        # Only run tests matching pattern
        suite = loader.discover("tests", pattern=pattern)
    else:
        # Run all tests
        suite = loader.discover("tests")
    
    # Set up test runner
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        failfast=failfast
    )
    
    # Run the tests
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="Run DualPipe tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pattern", "-p", help="Pattern to match test files")
    parser.add_argument("--failfast", "-f", action="store_true", help="Stop at first failure")
    
    args = parser.parse_args()
    
    return run_tests(
        verbose=args.verbose,
        pattern=args.pattern,
        failfast=args.failfast
    )


if __name__ == "__main__":
    sys.exit(main()) 
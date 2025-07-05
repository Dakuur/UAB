#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:: Labs of Graphs, Topology, and Discrete Geometry - Data Engineering, UAB - 2019/2020 ::

Run all test modules in the current working dir. 

NOTE: Students may run the ./test/test_lab0.py, ./test/test_lab1.py, etc., test scripts for each Lab assignment.
"""

import sys
import argparse
import os
import unittest
import datetime

# Add .. to the path so that tests can run as if run from the main dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Be verbose? Repeat for more",
                    action="count", default=0)
parser.add_argument("-f", "--full", help="Run full tests?",
                    action="count", default=0)
parser.add_argument("-x", "--dont_exit_on_error", help="Exit when an error is encountered in the tests?",
                    action="store_true")
parser.add_argument("--id", help="Your group ID", type=str, required=True)
options = parser.parse_known_args()[0]
assert options.id != "template", \
    "Template should be copied to labX_groupGG before testing"
assert options.id.lower() != "groupGG".lower(), \
    "You should substitute groupGG for your actual group name"

all_days = [
    "20191023", "20191024", "20191025", "20191026", "20191027", "20191028",
    "20191029", "20191030", "20191031", "20191101", "20191102", "20191103",
    "20191104", "20191105", "20191106", "20191107", "20191108", "20191109",
    "20191110", "20191111", "20191112", "20191113", "20191114", "20191115",
    "20191116", "20191117", "20191118", "20191119", "20191120", "20191121",
    "20191122"]

if __name__ == '__main__':
    suite = unittest.TestLoader().discover(os.path.dirname(__file__),
                                           pattern="test_*solution.py")

    if options.verbose:
        print(f"Running {suite.countTestCases()} tests @ {datetime.datetime.now()}")
        print(f"{'[Params]':-^30s}")
        for param, value in options.__dict__.items():
            print(f"{param}: {value}")
        print(f"{'':-^30s}")
        print()

    unittest.TextTestRunner(verbosity=3 if options.verbose else 1).run(suite)

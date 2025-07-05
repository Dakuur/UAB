#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:: Labs of Graphs, Topology, and Discrete Geometry - Data Engineering, UAB - 2019/2020 ::

Unit tests for the renfe.py module.
"""

import unittest
import random
import sys

random.seed(0xdeadbeef)

if __name__ == '__main__':
    sys.argv += ["--id", "test_renfe"]
import test_all
import renfe


class TestRenfeBasic(unittest.TestCase):
    number_days_tested = len(test_all.all_days) if test_all.options.full else 1

    def test_csv_read(self):
        if test_all.options.verbose > 1:
            print(f"\nTesting for {self.number_days_tested} days...")
        rg = renfe.RenfeBasicGrapher()
        for day in random.sample(test_all.all_days, self.number_days_tested):
            if test_all.options.verbose > 1:
                print(f"Testing for day {day}")
            G = rg.get_networkx_graph(target_day=day)
            assert 487 <= len(G.nodes) <= 1535, len(G.nodes)
            assert 487 <= len(G.edges) <= 567, len(G.edges)


if __name__ == '__main__':
    sys.argv = sys.argv[0:1] + [arg for arg in sys.argv if arg == "-v"]
    unittest.main()

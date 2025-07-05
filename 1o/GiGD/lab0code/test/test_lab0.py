#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:: Labs of Graphs, Topology, and Discrete Geometry - Data Engineering, UAB - 2019/2020 ::

Example (incomplete) unit tests for Lab0. Students are expected to add and submit their own test implementations. 

NOTE: new tests can be added to the TestLab subclass defined here, starting with def test_, accepting one parameter (self), and with the right indentation, e.g.,
   def test_new_test1(self):
        assert False
"""

import sys
import unittest

from test_lab import TestLab
import renfe


class TestLab0(TestLab, unittest.TestCase):
    name = "lab0"

    def test_create_graph_1(self):
        G = self.released_module.lab0.create_graph_1()
        assert sorted(G.nodes) == sorted(
            ['jake', 'charles', 'amy', 'gina', 'raymond'])

        valid_edges = set([("charles", "jake"), ("gina", "charles"),
                           ("jake", "raymond"), ("jake", "amy"), ("amy", "gina")])
        for edge in valid_edges:
            assert edge in G.edges, edge
            assert tuple(reversed(edge)) in G.edges, tuple(reversed(edge))
        for edge in G.edges:
            assert edge in valid_edges or tuple(reversed(edge)) in valid_edges, edge

    def test_create_graph_2(self):
        G = self.released_module.lab0.create_graph_2()
        edge_list = [(2, 3), (2, 5), (1, 2), (1, 10), (1, 20), (5, 6), (10, 5)]
        for edge in edge_list:
            assert edge in G.edges, f"{edge} not in G.edges, but it should"
            if tuple(reversed(edge)) not in edge_list:
                assert tuple(reversed(edge)) not in G.edges, \
                    f"{tuple(reversed(edge))} in G.edges, but should not"

    # def test_create_station_graph(self):
    #     G = self.released_module.lab0.create_station_graph()
    #     assert sum(G.edges[e]["count"] for e in G.edges) == 45825, \
    #         f"Are you counting every trip correctly?"
    #     reader = renfe.RenfeReader()
    #     stations_by_id = reader.get_stations_by_id()
    #     station_uab = stations_by_id['72503']
    #     station_cerdanyola = stations_by_id['78706']
    #     station_sant_cugat = stations_by_id['72502']
    #     assert G.edges[station_uab, station_cerdanyola]["count"] == 67
    #     assert G.edges[station_sant_cugat, station_uab]["count"] == 32


if __name__ == '__main__':
    sys.argv = sys.argv[0:1] + [arg for arg in sys.argv if arg == "-v"]
    unittest.main()

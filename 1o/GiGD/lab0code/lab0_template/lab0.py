#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:: Labs of Graphs, Topology, and Discrete Geometry - Data Engineering, UAB - 2019/2020 ::

Template for Lab0's tasks. See the assignment for details.

IMPORTANT: Don't change the provided function names, parameters or expected return.
"""

import networkx
import renfe

def create_empty_graph():
    return networkx.Graph()


def create_graph_1():
    # Empty graph
    G = networkx.Graph()
    # Add nodes
    G.add_node("jake")
    G.add_node("charles")
    G.add_node("amy")
    G.add_node("gina")
    G.add_node("raymond")
    # Add edges
    G.add_edge("charles", "gina")
    G.add_edge("charles", "jake")
    G.add_edge("amy", "jake")
    G.add_edge("amy", "gina")
    G.add_edge("raymond", "jake")
    # Return the graph
    return G


def create_graph_2():
    raise NotImplementedError()

def create_station_graph():
    raise NotImplementedError()
        



import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pandas as pd

def create_graph_1():
    graf = nx.primer(100,2)
    nx.draw_spring(graf)

    plt.show()
import networkx as nx
import random
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

regular_graph = nx.random_regular_graph(d=3, n=100)

clique_graph = nx.connected_caveman_graph(10, 20)

merged_graph = nx.union(regular_graph, clique_graph, rename=('G1-', 'G2-'))

for _ in range(10):
    node1 = random.choice(list(merged_graph.nodes))
    node2 = random.choice(list(merged_graph.nodes))
    if node1 != node2 and not merged_graph.has_edge(node1, node2):
        merged_graph.add_edge(node1, node2)

def calculate_egonet_features(G, node):
    egonet = nx.ego_graph(G, node)
    neighbors = list(egonet.nodes)
    neighbors.remove(node)
    Ni = len(neighbors)
    Ei = len(egonet.edges)
    Wi = sum([G[u][v].get('weight', 1) for u, v in egonet.edges])
    adj_matrix = nx.to_numpy_array(egonet, weight='weight')
    eigenvalues, _ = np.linalg.eigh(adj_matrix)
    lambda_w = max(eigenvalues)
    return Ni, Ei, Wi, lambda_w

features = {}
for node in merged_graph.nodes:
    Ni, Ei, Wi, lambda_w = calculate_egonet_features(merged_graph, node)
    features[node] = {'Ni': Ni, 'Ei': Ei, 'Wi': Wi, 'lambda_w': lambda_w}

Ni_Ei = np.array([[features[node]['Ni'], features[node]['Ei']] for node in merged_graph.nodes])
Ni_log = np.log(Ni_Ei + 1)
X, y = Ni_log[:, 0], Ni_log[:, 1]
X = X.reshape(-1, 1)

reg = LinearRegression().fit(X, y)
C, theta = reg.intercept_, reg.coef_[0]

scores = {}
for node in merged_graph.nodes:
    Ni = features[node]['Ni']
    Ei = features[node]['Ei']
    score = abs(Ei - (C * (Ni ** theta))) + 1
    scores[node] = score

top_10_nodes = sorted(scores, key=scores.get, reverse=True)[:10]

color_map = ['blue' if node not in top_10_nodes else 'red' for node in merged_graph.nodes]
nx.draw(merged_graph, node_color=color_map, with_labels=False, node_size=50)
plt.title("Graph with Top 10 Cliques Highlighted")
plt.show()
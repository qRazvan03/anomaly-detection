import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

file_path = 'ca-AstroPh.txt'
G = nx.Graph()

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i >= 1500:
            break
        node1, node2 = map(int, line.strip().split())
        if G.has_edge(node1, node2):
            G[node1][node2]['weight'] += 1
        else:
            G.add_edge(node1, node2, weight=1)

features = {}
for node in G.nodes:
    egonet = nx.ego_graph(G, node)
    Ni = len(egonet.nodes) - 1  # Number of neighbors (exclude the node itself)
    Ei = egonet.size(weight=None)  # Number of edges in the egonet
    Wi = egonet.size(weight="weight")  # Total weight of the egonet

    # Compute principal eigenvalue of the weighted adjacency matrix
    adj_matrix = nx.to_numpy_array(egonet, weight='weight')
    eigenvalues = np.linalg.eigvals(adj_matrix)
    lambda_w_i = max(eigenvalues)

    features[node] = {'Ni': Ni, 'Ei': Ei, 'Wi': Wi, 'lambda_w_i': lambda_w_i}

nx.set_node_attributes(G, features)

Ni = np.array([features[node]['Ni'] for node in G.nodes]).reshape(-1, 1)
Ei = np.array([features[node]['Ei'] for node in G.nodes]).reshape(-1, 1)

log_Ni = np.log(Ni + 1)
log_Ei = np.log(Ei + 1)

regressor = LinearRegression()
regressor.fit(log_Ni, log_Ei)
predicted_Ei = regressor.predict(log_Ni)

scores = []
for i, node in enumerate(G.nodes):
    yi = Ei[i, 0]
    Cxi_theta = predicted_Ei[i, 0]
    score = max(yi, Cxi_theta) / min(yi, Cxi_theta) * np.log(abs(yi - Cxi_theta) + 1)
    scores.append(score)

for i, node in enumerate(G.nodes):
    G.nodes[node]['score'] = scores[i]

sorted_nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['score'], reverse=True)
top_10_nodes = sorted_nodes[:10]

pos = nx.spring_layout(G)  
node_colors = ['red' if node in top_10_nodes else 'blue' for node in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50)
plt.title("Graph with Top 10 Anomalies Highlighted")
plt.show()

features_matrix = np.hstack((log_Ei, log_Ni))
lof = LocalOutlierFactor(n_neighbors=20)
lof_scores = -lof.fit_predict(features_matrix)  

final_scores = scores + lof_scores
for i, node in enumerate(G.nodes):
    G.nodes[node]['final_score'] = final_scores[i]

sorted_nodes_final = sorted(G.nodes, key=lambda x: G.nodes[x]['final_score'], reverse=True)
top_10_nodes_final = sorted_nodes_final[:10]

node_colors_final = ['green' if node in top_10_nodes_final else 'blue' for node in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw(G, pos, node_color=node_colors_final, with_labels=False, node_size=50)
plt.title("Graph with Top 10 Anomalies Highlighted (Using Final Scores)")
plt.show()
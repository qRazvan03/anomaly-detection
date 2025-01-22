import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def create_graph_with_anomalies():
    G = nx.random_regular_graph(d=3, n=100)

    anomaly_subgraph = nx.complete_graph(5)  
    G = nx.disjoint_union(G, anomaly_subgraph) 
    anomaly_nodes = list(range(100, 105))  

    for _ in range(10):
        u = np.random.choice(list(G.nodes))
        v = np.random.choice(list(G.nodes))
        if u != v:
            if G.has_edge(u, v):
                G[u][v]["weight"] = G[u][v].get("weight", 1) + 10 
            else:
                G.add_edge(u, v, weight=10)

    return G, anomaly_nodes

def extract_features(G):
    features = []
    for node in G.nodes:
        ego = nx.ego_graph(G, node)
        Ni = len(ego.nodes) - 1 
        Ei = len(ego.edges)  
        Wi = sum([G[u][v].get("weight", 1) for u, v in ego.edges])  
        features.append([Ni, Ei, Wi])
    return np.array(features)

def train_anomaly_detector(features):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    scores = -model.decision_function(features) 
    return model, scores

def evaluate_model(scores, true_anomalies, threshold=None):
    if threshold is None:
        threshold = np.percentile(scores, 90)  
    predicted_anomalies = scores > threshold
    y_true = np.zeros_like(scores, dtype=int)
    y_true[true_anomalies] = 1 

    precision = precision_score(y_true, predicted_anomalies)
    recall = recall_score(y_true, predicted_anomalies)
    f1 = f1_score(y_true, predicted_anomalies)
    return precision, recall, f1

G, true_anomalies = create_graph_with_anomalies()
features = extract_features(G)
model, scores = train_anomaly_detector(features)
precision, recall, f1 = evaluate_model(scores, true_anomalies)

node_colors = ["red" if i in true_anomalies else "blue" for i in range(len(G.nodes))]
nx.draw(G, node_color=node_colors, with_labels=False, node_size=50)
plt.title(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
plt.show()
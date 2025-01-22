import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.algorithms.clique import find_cliques

regular_graph = nx.random_regular_graph(d=3, n=100)

clique_graph = nx.connected_caveman_graph(10, 20)

merged_graph = nx.union(regular_graph, clique_graph, rename=('G1-', 'G2-'))

for _ in range(10): 
    node1 = random.choice(list(merged_graph.nodes))
    node2 = random.choice(list(merged_graph.nodes))
    if node1 != node2 and not merged_graph.has_edge(node1, node2):
        merged_graph.add_edge(node1, node2)

cliques = list(find_cliques(merged_graph)) 
largest_clique = max(cliques, key=len) 

color_map = []
for node in merged_graph.nodes:
    if node in largest_clique:
        color_map.append('red')  
    else:
        color_map.append('blue')  

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(merged_graph)  
nx.draw(merged_graph, pos, node_color=color_map, with_labels=False, node_size=50)
nx.draw_networkx_nodes(merged_graph, pos, nodelist=largest_clique, node_color='green', node_size=100)
nx.draw_networkx_edges(merged_graph, pos, edgelist=[(u, v) for u in largest_clique for v in largest_clique if merged_graph.has_edge(u, v)], edge_color='red', width=2)
plt.title("Graph Highlighting the Largest Clique")
plt.show()

print("Largest Clique Nodes:", largest_clique)
print("Size of the Largest Clique:", len(largest_clique))
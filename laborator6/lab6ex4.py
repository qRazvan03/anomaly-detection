import networkx as nx
import matplotlib.pyplot as plt
import random
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


node_clique_count = {} 
for clique in cliques:
    for node in clique:
        if node in node_clique_count:
            node_clique_count[node] += 1
        else:
            node_clique_count[node] = 1

overlapping_nodes = [node for node, count in node_clique_count.items() if count > 1]

color_map = []
for node in merged_graph.nodes:
    if node in overlapping_nodes:
        color_map.append('orange') 
    else:
        color_map.append('blue')  

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(merged_graph)  
nx.draw(merged_graph, pos, node_color=color_map, with_labels=False, node_size=50)
nx.draw_networkx_nodes(merged_graph, pos, nodelist=overlapping_nodes, node_color='red', node_size=100)
plt.title("Graph Highlighting Overlapping Nodes Between Cliques")
plt.show()

print("Overlapping Nodes:", overlapping_nodes)
print("Number of Overlapping Nodes:", len(overlapping_nodes))
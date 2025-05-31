import pandas as pd
from collections import defaultdict
import timeit
import random
import copy
import math
from math import radians, cos, sin, asin, sqrt
import queue
from queue import PriorityQueue
import heapq
from collections import deque
class Graph:
        def __init__(self):
            self.nodes = {}
            self.all_paths = {}
            
        def add_node(self, node, lat, lon):
            self.nodes[node] = {'latitude': lat, 'longitude': lon, 'neighbors': {}}
            self.all_paths[node] = {}
            return self

        def add_edge(self, node1, node2, cost=None):
            self.nodes[node1]['neighbors'][node2] = cost
            self.nodes[node2]['neighbors'][node1] = cost
            return self

        def delete_node(self, node):
            for neighbor in self.nodes[node]['neighbors']:
                self.delete_edge(node, neighbor)
            del self.nodes[node]
            return self

        def delete_edge(self, node1, node2):
            del self.nodes[node1]['neighbors'][node2]
            del self.nodes[node2]['neighbors'][node1]
            return self
        
        def get_neighbors(self, node):
                """
                Returns a dictionary of the neighboring vertices of the specified node and their edge weights.
                
                Args:
                    node: the node.
                
                Returns:
                    A dictionary of the neighboring vertices of the specified node and their edge weights.
                """
                if node not in self.nodes[node]:
                    return {}
                return self.edges[node]

city_graph = Graph()        #   GRAPH THAT CAN BE REPLICATED
        
with open('coordinates.txt', 'r') as file:
    next(file)  # skip first line
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            city, latitude, longitude = parts
            city_graph.add_node(city, float(latitude), float(longitude))
        elif len(parts) > 3:
            city = " ".join(parts[:-2])
            latitude, longitude = parts[-2:]
            city_graph.add_node(city, float(latitude), float(longitude))






'''
ALL ALGORITHMS DEFINED HERE:
'''


#   DFS

def dfs(start, goal, graph):
    visited = set()
    stack = [start]
    total_path = []
    while stack:
        current = stack.pop()
        neighbors = graph.nodes[current]['neighbors']
        if current not in visited:
            visited.add(current)
            total_path.append(current)
            if current == goal:
                return total_path
            for neighbor in neighbors:
                stack.append(neighbor)

#   BFS

def bfs(start, goal, graph):
    visited = set()
    queue = [start]
    total_path = []
    while queue:
        current = queue.pop(0)
        neighbors = graph.nodes[current]['neighbors']
        if current not in visited:
            visited.add(current)
            total_path.append(current)
            if current == goal:
                return total_path
            for neighbor in neighbors:
                queue.append(neighbor)


def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def h(current, goal):
    #Longitude and Latitude Heuristic Function
    current_node = current
    current_latitude = graph.nodes[current_node]['latitude']
    current_longitude = graph.nodes[current_node]['longitude']

    goal_node = goal
    goal_latitude = graph.nodes[goal_node]['latitude']
    goal_longitude = graph.nodes[goal_node]['longitude']

    h = math.sqrt((goal_longitude - current_longitude)**2 + (goal_latitude - current_latitude)**2)
    return h


#   A STAR

def a_star(start, goal, graph):
    final_path = [[],[],[]]
    priority_queue = [(h(start, goal), start, [start])]
    while priority_queue:
        
        current_f, current, path = heapq.heappop(priority_queue)
        targeted = (current_f, current, path)
    
        current_g = current_f - h(current, goal)        # REAL COST SO FAR
        
        neighbors = graph.nodes[current]['neighbors']
        if current == goal:
            
            final_path = current_f, current, path
            break
        
        for neighbor in neighbors:
            neighbor_f = current_f +graph.nodes[current]['neighbors'][neighbor] + h(neighbor, goal)
            heapq.heappush(priority_queue, (neighbor_f, neighbor, path+[neighbor]))
    
    return final_path[2]

#bidirectional_search

def bidirectional_search(start_node, end_node, graph):
        forward_frontier = deque([(start_node, [start_node])])
        backward_frontier = deque([(end_node, [end_node])])
        forward_explored = {start_node}
        backward_explored = {end_node}
        backward_path=[]

        while forward_frontier and backward_frontier:
            current_forward_node, forward_path = forward_frontier.popleft()
            for neighbor in graph.nodes[current_forward_node]['neighbors']:
                if neighbor not in forward_explored:
                    new_forward_path = forward_path + [neighbor]
                    forward_frontier.append((neighbor, new_forward_path))
                    forward_explored.add(neighbor)
                    if neighbor in backward_explored:
                        # We've found a path from start to end!
                        return new_forward_path + backward_path[::-1]

            current_backward_node, backward_path = backward_frontier.popleft()
            for neighbor in graph.nodes[current_backward_node]['neighbors']:
                if neighbor not in backward_explored:
                    new_backward_path = backward_path + [neighbor]
                    backward_frontier.append((neighbor, new_backward_path))
                    backward_explored.add(neighbor)
                    if neighbor in forward_explored:
                        # We've found a path from start to end!\
                        return forward_path + new_backward_path[::-1]


#       UCS
def ucs(start_node, goal_node, graph):
    frontier = [(0, start_node)]        
    cost_so_far = {start_node: 0}
    path_so_far = {start_node: []}

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal_node:
            return path_so_far[current_node]

        for neighbor, cost in graph.nodes[current_node]['neighbors'].items():
            new_cost = cost_so_far[current_node] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                path_so_far[neighbor] = path_so_far[current_node] + [(current_node, neighbor)]
                heapq.heappush(frontier, (new_cost, neighbor))


                
    # Add edges with costs
city_graph.add_edge('Oradea', 'Zerind', cost=71)
city_graph.add_edge('Oradea', 'Sibiu', cost=151)
city_graph.add_edge('Zerind', 'Arad', cost=75)
city_graph.add_edge('Arad', 'Sibiu', cost=140)
city_graph.add_edge('Arad', 'Timisoara', cost=118)
city_graph.add_edge('Timisoara', 'Lugoj', cost=111)
city_graph.add_edge('Lugoj', 'Mehadia', cost=70)
city_graph.add_edge('Mehadia', 'Drobeta', cost=75)
city_graph.add_edge('Drobeta', 'Craiova', cost=120)
city_graph.add_edge('Craiova', 'Rimnicu Vilcea', cost=146)
city_graph.add_edge('Craiova', 'Pitesti', cost=138)
city_graph.add_edge('Rimnicu Vilcea', 'Sibiu', cost=80)
city_graph.add_edge('Rimnicu Vilcea', 'Pitesti', cost=97)
city_graph.add_edge('Sibiu', 'Fagaras', cost=99)
city_graph.add_edge('Fagaras', 'Bucharest', cost=211)
city_graph.add_edge('Pitesti', 'Bucharest', cost=101)
city_graph.add_edge('Bucharest', 'Giurgiu', cost=90)
city_graph.add_edge('Bucharest', 'Urziceni', cost=85)
city_graph.add_edge('Urziceni', 'Hirsova', cost=98)
city_graph.add_edge('Hirsova', 'Eforie', cost=86)
city_graph.add_edge('Urziceni', 'Vaslui', cost=142)
city_graph.add_edge('Vaslui', 'Iasi', cost=92)
city_graph.add_edge('Neamt', 'Iasi', cost=87)







'''
CITY GRAPH TEST
'''

#   duplicating city graph and testing it
graph = copy.deepcopy(city_graph)       
                    

#randomly pick 10 cities

pick = random.sample(list(graph.nodes),k = 10)

time_taken = 0
length = 0
count = 0
analysis = defaultdict(list)

functions = [dfs,bfs,a_star,bidirectional_search, ucs]



while count < 10:
    
    start = random.choices(list(graph.nodes),k = 1)[0]
    goal = random.choices(list(graph.nodes),k = 1)[0]

    
    if start == goal:
        continue
    
    for func in functions:
        total = 0
        total_length = 0
        
    
        for _ in range(10): 
            begin = timeit.default_timer()
            result = func(start,goal,graph)
            finish = timeit.default_timer()
            
            
            if result:
                length = len(result)
                total_length += length
    
            execution_time = (finish - begin) * 1000
            total += execution_time
        average_time = total / 10
        average_time = round(average_time,3)
        average_length = total_length // 10
        analysis[func.__name__].append([average_length,average_time])
    count+=1
    
for algorithm, pair in analysis.items():
    print(algorithm, pair)


import pandas as pd

# create a list of dictionaries representing each row in the table
rows = []
for algorithm in analysis:
    for algorithm_index, (length, time) in enumerate(analysis[algorithm]):
        rows.append({
            'Start': start,
            'Goal': goal,
            'Algorithm': algorithm,
            'Average Length': length,
            'Average Time': time
        })

# create a Pandas DataFrame from the list of dictionaries
df = pd.DataFrame(rows)

# display the DataFrame
print("TABLE")
print(df)

#   ALL CITY GRAPH TESTS DONE.







'''
16 RANDOM GRAPHS TEST
'''

# Create 16 random graphs with different node sizes and edge probabilities
analysis = defaultdict(list)
node_sizes = [10, 20, 30, 40]

edge_probs = [0.2, 0.4, 0.6, 0.8]

graphs = {}
for node_size in node_sizes:
    for probability in edge_probs:
        # Create a new Graph object
        graph = Graph()
        
        # Add nodes to the graph
        for i in range(node_size):
            graph.add_node(i, random.uniform(0, 1), random.uniform(0, 1))
        
        # Add edges to the graph with probability p
        for i in range(node_size):
            for j in range(i+1, node_size):
                if random.random() < probability:
                    graph.add_edge(i, j, random.randint(1, 10))

        graphs[graph] = [node_size, probability]





for graph in graphs:
    count = 0
    while count < 5:
        
        start = random.choices(list(graph.nodes),k = 1)[0]
        goal = random.choices(list(graph.nodes),k = 1)[0]
        
        if start == goal:
            continue
        
        for func in functions:
            total = 0
            total_length = 0
            
            for _ in range(10): 
                begin = timeit.default_timer()
                result = func(start,goal,graph)
                finish = timeit.default_timer()
                
                if result:
                    length = len(result)
                    total_length += length
        
                execution_time = (finish - begin) * 1000
                
                total += execution_time
            average_time = total / 10
            average_time = round(average_time,3)
            average_length = total_length // 10
            analysis[func.__name__].append([average_length,average_time])
        count+=1
        
for algorithm, pair in analysis.items():
    print(algorithm, pair)



'''

import matplotlib.pyplot as plt

# Create a dictionary to map edge probabilities to colors
edge_prob_colors = {0.2: 'blue', 0.4: 'orange', 0.6: 'green', 0.8: 'red'}

# Create a list of markers to use for each algorithm
markers = ['o', 's', 'D', '^', 'v']

# Create a scatter plot for each algorithm
for i, func in enumerate(functions):
    print(i)
    plt.figure(i)
    plt.title(func.__name__)
    plt.xlabel('Average Execution Time (ms)')
    plt.ylabel('Average Path Length')
    for graph, params in graphs.items():
        node_size, edge_prob = params
        color = edge_prob_colors[edge_prob]
        marker = markers[i]
        key = f'{graph.nodes[0]} --> {graph.nodes[1]}'
        data = analysis[key][i]
        avg_length, avg_time = data
        label = f'n={node_size}, p={edge_prob}'
        plt.scatter(avg_time, avg_length, c=color, marker=marker, label=label)
    plt.legend()

plt.show()
'''

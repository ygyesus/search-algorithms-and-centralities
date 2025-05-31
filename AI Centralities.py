import heapq
import copy

import random
import math
import numpy as np

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

city_graph = Graph()
    
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

graph = copy.deepcopy(city_graph) 

path = []
            
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


def a_star(start, goal):
    priority_queue = [(h(start, goal), start, [start])]
    while priority_queue:
        current_f, current, path = heapq.heappop(priority_queue)
        targeted = (current_f, current, path)
    
        current_g = current_f - h(current, goal)        # REAL COST SO FAR
        
        neighbors = graph.nodes[current]['neighbors']
        if current == goal:
            
            return current_f, current, path
            break
        
        for neighbor in neighbors:
            neighbor_f = current_f +graph.nodes[current]['neighbors'][neighbor] + h(neighbor, goal)
            heapq.heappush(priority_queue, (neighbor_f, neighbor, path+[neighbor]))

matrix = dict()
visited_pairs = set()
paths = dict()
for node in graph.nodes:
    matrix[node] = dict()    
    for other_node in graph.nodes:
        if (node, other_node) not in visited_pairs and (other_node, node) not in visited_pairs:
                visited_pairs.add((node, other_node))
                total_cost = 0
                if node == other_node:
                    
                    matrix[node][other_node] = 0
                    
                    
                else:
                    targeted = a_star(node, other_node)
                    cost = targeted[0]
                    path = targeted[2]
                    matrix[node][other_node] = cost
                    
                    for next_index in range (1, len(path)):
                        next = path[next_index]
                        now = path[next_index-1]
                        total_cost += graph.nodes[now]['neighbors'][next]
                
                        
                if node not in graph.nodes[other_node]['neighbors']:
                    paths[(node, other_node)] = path
                        
          
                
            

adjacency_matrix = [[0 for i in range(len(graph.nodes))]for j in range (len(graph.nodes))]

node_to_index = dict()
matrix_index = 0
row_of_nodes = []
for node in graph.nodes:
    node_to_index[node] = matrix_index
    matrix_index += 1
    row_of_nodes.append(node)
for node in graph.nodes:
    i = node_to_index[node]
    for other_node in graph.nodes:
        j = node_to_index[other_node]   
        if other_node in graph.nodes[node]['neighbors']:
            adjacency_matrix[node_to_index[node]][node_to_index[other_node]] = graph.nodes[node]['neighbors'][other_node]

graph.adjacency_matrix = np.array(adjacency_matrix)





#   DEGREE CENTRALITY

def degree(graph):
    adjacency_matrix = graph.adjacency_matrix

    total_graph_cost = 0
    for i in range (adjacency_matrix.shape[0]):
        for j in range (adjacency_matrix.shape[1]):
            total_graph_cost += adjacency_matrix[i][j]

    total_graph_cost /= 2
    degree_dict = {}
    for city in graph.nodes:    #EACH NODE
        neighbors = graph.nodes[city]['neighbors']
        sum = 0
        number_of_edges = 0
        for other_city, cost in neighbors.items():      # FOR CITY'S COST WITH EACH NEIGHBOR
            if city != other_city:
                sum += cost
                number_of_edges += 1

        
        degree_dict[city] = sum/total_graph_cost

    my_list = []
    for key, value in degree_dict.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)
        
    return my_list


#   CLOSENESS CENTRALITY
                
def closeness(graph):
    
    closeness = dict()

    for node in graph.nodes:
        
        total_cost = 0
        
        for other_node in graph.nodes:
            cost = a_star(node, other_node)[0]
            
            total_cost += cost
            
        
        closeness[node] = (len(graph.nodes)-1)/(total_cost)            

    my_list = []
    for key, value in closeness.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)
    
    return my_list


#   EIGEN CENTRALITY

def eigen(graph):
    adjacency_matrix = graph.adjacency_matrix
    col_vector = np.array([1 for x in range (len(graph.nodes))])
    
    while True:
        earlier_col_vector = col_vector
        col_vector = np.dot(adjacency_matrix, col_vector)

        # NORMALIZING COL_VECTOR
        sum = 0
        for element in col_vector:
            sum += element**2
        
        normalized_value = sum**0.5
        col_vector = col_vector / normalized_value
        
        # EUCLIDEAN DISTANCE BETWEEN TWO VECTORS

        diff = [col_vector[i] - earlier_col_vector[i] for i in range (len(col_vector))]
        euclidean_distance = 0
        
        for difference in diff:
            euclidean_distance += difference**2
        euclidean_distance = euclidean_distance**0.5
        
        if euclidean_distance < 1e-06:    
            break
        
    eigen_dict = dict()
    
    for centrality_index in range(len(col_vector)):
        eigen_centrality = col_vector[centrality_index]
        node = row_of_nodes[centrality_index]
        eigen_dict[node] = eigen_centrality

    my_list = []
    for key, value in eigen_dict.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)

    return my_list

#   KATZ CENTRALITY

def katz(graph, alpha=0.4, beta=0.2, tol=0.01):
    col_vector = np.array([1 for x in range (len(graph.nodes))])
    
    while True:
        
        earlier_col_vector = col_vector
        col_vector = alpha * np.dot(adjacency_matrix, col_vector) + beta * np.array([1 for x in range (len(graph.nodes))])

        # NORMALIZING COL_VECTOR
        sum = 0
        for element in col_vector:
            sum += element**2
        
        normalized_value = sum**0.5
        col_vector = col_vector / normalized_value
        
        # EUCLIDEAN DISTANCE BETWEEN TWO VECTORS

        diff = [col_vector[i] - earlier_col_vector[i] for i in range (len(col_vector))]
        euclidean_distance = 0
        
        for difference in diff:
            euclidean_distance += difference**2
        euclidean_distance = euclidean_distance**0.5
        
        if euclidean_distance < tol:    
            break
        
    katz_dict = dict()
    for centrality_index in range(len(col_vector)):
        katz_centrality = col_vector[centrality_index]
        node = row_of_nodes[centrality_index]
        katz_dict[node] = katz_centrality

    
    my_list = []
    for key, value in katz_dict.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)

    return my_list




# PAGERANK CENTRALITY

def page_rank(graph, alpha=0.1, tol=400):
    adjacency_matrix = graph.adjacency_matrix
    col_vector = np.array([1 for num in range(len(graph.nodes))])
    ones = col_vector

    page_rank_dict = dict()
    while True:
        earlier_col_vector = col_vector
        col_vector = alpha * np.dot(adjacency_matrix, earlier_col_vector) + ones*(1-alpha)/len(graph.nodes)

        diff = [col_vector[i] - earlier_col_vector[i] for i in range (len(col_vector))]
        euclidean_distance = 0

        for difference in diff:
            
            euclidean_distance += difference**2
            
        euclidean_distance = euclidean_distance**0.5
        
        if euclidean_distance < tol:
            break
        
    col_vector = col_vector/sum(col_vector)
    for centrality_index in range(len(col_vector)):
        node = row_of_nodes[centrality_index]
        page_rank_centrality = col_vector[centrality_index]    
        page_rank_dict[node] = page_rank_centrality

    my_list = []
    for key, value in page_rank_dict.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)
        
    return my_list
    


#   BETWEENNESS CENTRALITY


def betweenness(graph):
    betweenness_dict = {}
    for node in graph.nodes:
        betweenness_dict[node] = 0
        
    for tuple in paths:
        path = paths[tuple][1:-1]
        
        for between_node in path:
            
            betweenness_dict[between_node] += 1
    

    my_list = []
    for key, value in betweenness_dict.items():
        my_list.append((value, key))

    my_list.sort(reverse=True)
    
    return my_list


'''
TEST
'''




#   DEGREE
print (" ")
print ("DEGREE CENTRALITY")
for city in degree(graph):
    print(city)

    
#   CLOSENESS
print (" ")
print ("CLOSENESS CENTRALITY")
for city in closeness(graph):
    print(city)


#   EIGEN
print(" ")
print("EIGEN CENTRALITY")
for city in eigen(graph):
    print(city)


#   KATZ
print (" ")
print("KATZ CENTRALITY")
for city in katz(graph):
    print(city)

#   PAGERANK
print (" ")
print("PAGERANK CENTRALITY")
for city in page_rank(graph):
    print(city)

#   BETWEENNESS
print(" ")
print("BETWEENNESS CENTRALITY")
for city in betweenness(graph):
    print(city)






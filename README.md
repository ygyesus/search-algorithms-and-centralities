# Graph Search Algorithms and Network Analysis Implementation

This project implements various graph search algorithms and network analysis techniques as part of the Introduction to AI course assignment. The implementation focuses on graph operations, search algorithms, and centrality measures using a real-world dataset of Romanian cities.

## Project Overview

This project demonstrates the implementation and benchmarking of various graph search algorithms and network analysis techniques. The implementation uses a real-world dataset of Romanian cities with their geographical coordinates and road connections, as specified in the textbook (page 83).

## Objectives

1. Graph Operations and Search Algorithms
   - Implementation of basic and advanced search algorithms
   - Benchmarking and comparison of different search algorithms
   - Analysis of algorithm performance across different graph sizes and densities

2. Network Analysis
   - Implementation of various centrality measures
   - Analysis of node importance in the network
   - Comparison of different graph representations

## Features

### Search Algorithms
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- A* Search Algorithm with Haversine distance heuristic
- Uniform Cost Search (UCS)
- Bidirectional Search

### Network Analysis
- Centrality Measures:
  - Degree Centrality
  - Closeness Centrality
  - Eigenvector Centrality
  - Katz Centrality
  - PageRank
  - Betweenness Centrality
- Graph-based pathfinding
- Haversine distance calculation for geographical coordinates
- Adjacency matrix representation
- Cost-based edge weights

## Performance Analysis

### Search Algorithm Performance
The algorithms were tested across multiple paths with the following results. Each experiment was run 10 times to ensure statistical significance:

#### Path Lengths
- DFS: 2, 3, 4, 5, 6, 6, 8, 10, 16, 4 nodes
- BFS: 3, 5, 6, 7, 9, 13, 14, 16, 20, 3 nodes
- A*: 2, 2, 3, 3, 3, 5, 5, 5, 6, 6 nodes
- Bidirectional Search: 2, 2, 3, 3, 3, 4, 5, 5, 6, 6 nodes
- UCS: 1, 2, 2, 2, 4, 4, 5, 5, 4, 5 nodes

#### Time Complexity (seconds)
- DFS: 0.006, 0.014, 0.017, 0.03, 0.004, 0.049, 0.015, 0.015, 0.01, 0.004
- BFS: 0.005, 0.013, 0.011, 0.014, 0.023, 0.028, 0.04, 0.039, 0.105, 0.005
- A*: 0.041, 0.031, 0.045, 0.143, 0.122, 0.648, 0.916, 0.649, 3.487, 0.13
- Bidirectional Search: 0.006, 0.044, 0.007, 0.008, 0.008, 0.011, 0.013, 0.022, 0.016, 0.005
- UCS: 0.009, 0.037, 0.029, 0.026, 0.067, 0.055, 0.06, 0.051, 0.064, 0.019

### Random Graph Analysis
The algorithms were also tested on randomly generated graphs with varying sizes and edge probabilities:
- Graph sizes (n): 10, 20, 30, 40 nodes
- Edge probabilities (p): 0.2, 0.4, 0.6, 0.8
- Each experiment was run 5 times to ensure statistical significance
- Nodes were assigned random x,y coordinates for heuristic calculations

### Graph Representation Comparison
The project implements and compares three different graph representations:
1. Adjacency List
2. Adjacency Matrix
3. Edge List

Each representation was evaluated based on:
- Time complexity for:
  - Vertex insertion/deletion
  - Edge insertion/deletion
  - Edge existence checking
  - Neighbor finding
- Space complexity for each operation

### Centrality Analysis Results

The analysis of different centrality measures revealed that higher ranks are populated by roughly the same cities, with some differences in rankings:

#### Top 5 Cities by Centrality Measure

1. **Degree Centrality**
   - Bucharest
   - Sibiu
   - Craiova
   - Pitesti
   - Arad
   *Cities like Bucharest, Sibiu, and Craiova hold the top three ranks mainly due to their higher number of edges or immediate neighbors.*

2. **Closeness Centrality**
   - Pitesti
   - Rimnicu Vilcea
   - Bucharest
   - Sibiu
   - Craiova
   *These cities score highest because they are generally "closer" to other cities, with low overall cost incurred when connecting to all other nodes.*

3. **Eigenvector Centrality**
   - Bucharest
   - Pitesti
   - Fagaras
   - Craiova
   - Sibiu
   *Cities like Bucharest and Pitesti have higher eigenvector centralities as they are connected to other important cities, both directly and indirectly.*

4. **Katz Centrality**
   - Bucharest
   - Pitesti
   - Fagaras
   - Sibiu
   - Craiova
   *These cities have the highest katz centralities as many paths from one part of the graph to the other have to go through them.*

5. **PageRank Centrality**
   - Bucharest
   - Sibiu
   - Craiova
   - Pitesti
   - Arad
   *These cities are top scorers because of the higher numbers of important cities that link towards them.*

6. **Betweenness Centrality**
   - Bucharest
   - Urziceni
   - Pitesti
   - Riminicu Vilcea
   - Sibiu
   *When connecting any pair of cities using the shortest possible route, these cities are most likely to appear in the path.*

## Project Structure

- `Search Algorithms.py`: Contains implementations of various search algorithms
- `AI Centralities.py`: Implements network centrality measures and analysis
- `coordinates.txt`: Contains geographical coordinates data for Romanian cities
- `Summary.pdf`: Project documentation and analysis

## Implementation Details

The project uses a custom `Graph` class that supports:
- Node addition and deletion
- Edge management with weighted connections
- Geographical coordinates (latitude/longitude)
- Neighbor relationships
- Path finding
- Adjacency matrix representation
- All-pairs shortest paths calculation

### Key Components

1. **Graph Class**
   - Supports weighted and unweighted graphs
   - Handles geographical coordinates
   - Manages node and edge operations
   - Maintains adjacency matrix for centrality calculations
   - Supports bidirectional edges with costs

2. **Search Algorithms**
   - DFS: Depth-first traversal
   - BFS: Breadth-first traversal
   - A*: Heuristic-based search using Haversine distance
   - UCS: Cost-optimized path finding
   - Bidirectional Search: Two-way path finding

3. **Distance Calculation**
   - Implements Haversine formula for geographical distances
   - Supports coordinate-based path optimization
   - Calculates real-world distances between cities

4. **Centrality Measures**
   - Degree Centrality: Measures node importance based on connections
   - Closeness Centrality: Measures average distance to all other nodes
   - Eigenvector Centrality: Considers importance of connected nodes
   - Katz Centrality: Accounts for both direct and indirect connections
   - PageRank: Measures node importance in directed graphs
   - Betweenness Centrality: Measures node's role in shortest paths

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - collections
  - math
  - queue
  - heapq
  - numpy
  - random
  - matplotlib.pyplot (for visualization)

## Dataset

The project uses a dataset of Romanian cities including:
- City names
- Geographical coordinates (latitude/longitude)
- Road connections with distances
- Weighted edges representing real-world distances

## Contributors

- UGR/9155/13
- UGR/7847/13
- UGR/8932/13
- UGR/5887/13
- UGR/6364/13

## License

This project is part of a university assignment and is intended for educational purposes. 
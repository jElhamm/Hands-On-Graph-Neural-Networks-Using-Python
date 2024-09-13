# Chapter 2

   This chapter delves into foundational graph operations, visualization techniques, and analysis, all implemented using popular libraries like [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/).
   In this chapter, we introduce several important concepts related to graph theory and how these can be applied in Python.

## Key Concepts

   1. Graph Visualization:

      - Visualization of undirected, directed, and weighted graphs.
   
      - Usage of spring_layout and other layout algorithms for plotting.
   
      - Display of edge weights for weighted graphs.

   2. Graph Operations:

      - [Breadth-First Search (BFS)](https://en.wikipedia.org/wiki/Breadth-first_search) and [Depth-First Search (DFS)](https://en.wikipedia.org/wiki/Depth-first_search) traversals of a graph.

      - Checking the connectivity of graphs using the is_connected method.

   3. Graph Centrality Measures:

   - Calculation and display of degree centrality, closeness centrality, and betweenness centrality for a simple graph.

   4. Graph Representations:

   - Demonstrations of adjacency matrix and adjacency list representations of a graph.

## Code Highlights

   - **Graph Definitions**: Several types of graphs are defined, including an undirected graph, a directed graph, and a weighted graph.

   - **Visualization**: The GraphVisualizer class is used to visualize the defined graphs in a variety of layouts.

   - **Graph Connectivity**: The GraphAnalyzer checks the connectivity of two example graphs.

   - **Centrality Measures**: The centrality measures of the simple graph are printed, helping to assess the relative importance of nodes.

   - **Graph Traversal**: The BFS and DFS results for the simple graph are displayed.

   - **Adjacency Representations**: Both adjacency matrix and adjacency list representations are printed to illustrate different ways of representing a graph.

## Files And Run

   1. Python Script `(chapter2.py)`: You can run this script in your terminal or Python environment. Make sure you have NetworkX and Matplotlib installed:

```bash
   pip install networkx matplotlib
   python chapter2.py
```

   2. Jupyter Notebook `(chapter2.ipynb)`: For an interactive experience, open the notebook in Jupyter Notebook or JupyterLab:

```bash
   pip install networkx matplotlib
   jupyter notebook chapter2.ipynb
```

## Conclusion

   - This chapter serves as an essential building block for understanding graph structures and their analysis.
   
   - The visual and algorithmic insights gained here will be invaluable as we progress into more advanced topics like Graph Neural Networks in future chapters.

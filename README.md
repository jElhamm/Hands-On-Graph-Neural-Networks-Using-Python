<table>
  <tr>
    <td>
      <img src="https://m.media-amazon.com/images/I/61nGoPcI5jL._SY466_.jpg" alt="Hands-On Graph Neural Networks Book Cover" style="width: 2050px;"/>
    </td>
    <td>
      <h2>Hands-On Graph Neural Networks Using Python</h2>
      <p>Welcome to the complete code implementation for the book <strong>Hands-On Graph Neural Networks Using Python</strong>. This repository contains all the code examples from the book, organized into chapters for easy navigation, with each chapter provided in both `.py` and `.ipynb` formats. A `README.md` file accompanies each chapter to guide users through the respective code implementations. This repository is an excellent resource for learners, researchers, and developers interested in exploring and building powerful graph neural networks.</p>
    </td>
  </tr>
</table>


## 🚀 Quick Overview

   - **Author:** [Maxime Labonne](https://www.amazon.com/stores/Maxime-Labonne/author/B0BVKVJSSQ?ref=ap_rdr&isDramIntegrated=true&shoppingPortalEnabled=true)
   - **Focus Areas:** Graph Neural Networks, PyTorch Geometric, Machine Learning, Node Classification, Graph Embedding, and more
   - **Book Reference:** *Hands-On Graph Neural Networks Using Python*
   - - **Repository Contents:** 17 Chapters | `.py` and `.ipynb` files | Detailed chapter-wise `README.md` guides
   - **Download and buy the book:** [www.amazon.com](https://www.amazon.com/Hands-Graph-Neural-Networks-Python/dp/1804617520)

---

## ⭐️ Book Description

   [Graph Neural Networks (GNNs)](https://en.wikipedia.org/wiki/Graph_neural_network) have quickly emerged as a cutting-edge technology in deep learning, only a decade after their inception. They are transforming industries worth billions, such as drug discovery, where they played a pivotal role in predicting a novel antibiotic named Halicin. Today, tech companies are exploring their applications in various fields, including recommender systems for food, videos, and romantic partners, as well as fake news detection, chip design, and 3D reconstruction.

   In this book, "Graph Neural Networks," we will delve into the core principles of graph theory and learn how to create custom datasets from raw or tabular data. We will explore key graph neural network architectures to grasp essential concepts like graph convolution and self-attention. This foundational knowledge will then be used to understand and implement specialized models tailored for specific tasks such as link prediction and graph classification, as well as various contexts including spatio-temporal data and heterogeneous graphs. Ultimately, we will apply these techniques to solve real-world problems and begin building a professional portfolio.

---

## 📂 Repository Structure

   Here’s a summary of the chapters implemented in this repository, along with a brief description of each:

   | Chapter | Title                                        | Description |
   |---------|----------------------------------------------|-------------|
   | [01](main/Chapter01)      | Getting Started with Graph Learning          | Learn the basics of graph learning and graph neural networks (GNNs), and understand how to set up your first graph-based model. |
   | [02](main/Chapter02)       | Graph Theory for Graph Neural Networks       | Dive into essential graph theory concepts that form the backbone of GNNs and understand their relevance to deep learning models. |
   | [03](main/Chapter03)       | Creating Node Representations with DeepWalk  | Learn how to create node embeddings using the DeepWalk algorithm, transforming graph nodes into feature vectors for machine learning. |
   | [04](main/Chapter04)       | Node2Vec                                     | Implement the Node2Vec algorithm to generate improved node embeddings through biased random walks. |
   | [05](main/Chapter05)       | Vanilla Neural Network                       | Build and understand a basic, fully connected neural network to apply on graph data as a foundation for more complex GNN architectures. |
   | [06](main/Chapter06)       | Normalizing Features with Graph Convolutional Networks | Implement Graph Convolutional Networks (GCNs) to normalize node features and learn how to apply convolutional operations on graph data. |
   | [07](main/Chapter07)       | Graph Attention Network                      | Introduce attention mechanisms in graph learning through Graph Attention Networks (GAT) for enhanced performance on node classification tasks. |
   | [08](main/Chapter08)       | Scaling Graph Neural Networks                | Learn techniques to scale GNNs for large-scale graph data, ensuring efficient training on massive datasets. |
   | [09](main/Chapter09)       | Graph Classification                         | Implement GNNs for graph classification tasks, using real-world datasets to classify entire graphs rather than individual nodes. |
   | [10](main/Chapter10)       | Link Prediction                              | Use GNNs to predict links in graphs, helping identify missing or future connections between nodes. |
   | [11](main/Chapter11)       | Graph Generation                             | Explore generative models for graphs, learning how to create new graphs and complete partial ones using GNNs. |
   | [12](main/Chapter02)       | Learning from Heterogeneous Graphs           | Understand how to work with heterogeneous graphs and implement models like Heterogeneous Attention Networks (HAN) to process different types of nodes and edges. |
   | [13](main/Chapter13)       | Temporal Graph Neural Networks               | Learn to work with dynamic or temporal graphs and build GNNs that can handle evolving data over time. |
   | [14](main/Chapter14)       | Explainability                               | Implement GNNExplainer and other tools to interpret GNN models and make sense of the learned representations and predictions. |
   | [15](main/Chapter15)       | Traffic Forecasting                          | Use GNNs to forecast traffic patterns and other spatio-temporal data, improving decision-making in real-time applications. |
   | [16](main/Chapter16)       | Anomaly Detection                            | Apply GNNs to detect anomalies in graph data, identifying unusual patterns in networks such as fraud detection or outlier nodes. |
   | [17](main/Chapter17)       | Recommender Systems                          | Build recommender systems using GNNs to provide personalized recommendations for users, applying GNNs to problems like product, movie, or partner recommendations. |

---

## 💡 Prerequisites

Before running the code, make sure you have the following tools and libraries installed:

<table>
  <tr>
    <td>
      <ul>
        <li><strong>Python</strong></li>
        <li><strong>PyTorch</strong></li>
        <li><strong>PyTorch Geometric</strong></li>
        <li><strong>NetworkX</strong></li>
      </ul>
    </td>
    <td>
      <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="100" style="margin-right: 20px;"/>
      <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="PyTorch Logo" width="100" style="margin-right: 20px;"/>
      <img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true" alt="PyTorch Geometric Logo" width="100" style="margin-right: 20px;"/>
      <img src="https://avatars.githubusercontent.com/u/388785?s=200&v=4" alt="NetworkX Logo" width="100" style="margin-right: 20px;"/>
    </td>
  </tr>
</table>

<p>For each chapter of this repository, there may be additional libraries required. Please refer to the specific <code>README.md</code> files in each chapter directory to find detailed information about the libraries needed and installation instructions.</p>

---

## 💡 Running the Code

   Each chapter folder contains:

   - A Python script (`.py file`)
   - A Jupyter Notebook (`.ipynb` file)
   - A `README.md` with instructions and explanations for that specific chapter

   You can either run the Python scripts directly or execute the Jupyter notebooks interactively:

```bash
   # Run the Python script
   python chapter_x/script_name.py

   # Or, open the Jupyter notebook
   jupyter notebook chapter_x/notebook_name.ipynb
```
---
## ✨ Features

   - Learn the fundamentals of graph theory for data science and machine learning

   - Implement state-of-the-art graph neural network architectures

   - Build creative and powerful applications in various fields

   - Real-World Applications: Learn how to apply GNNs to various real-world problems such as molecular graphs, recommendation systems, and social networks.

   - Diverse GNN Architectures: Explore implementations of GCN, GAT, GraphSAGE, VGAE, HAN, and more.

   - Cutting-Edge Techniques: Learn about dynamic graphs, heterogeneous graphs, explainability in GNNs, and large-scale GNNs.

---
## 🤝 Contribution

   Contributions are welcome! If you find any issues or want to add improvements to the code, feel free to submit a pull request or open an issue.

   1. Fork the repository

   2. Create your feature branch (`git checkout -b feature/AmazingFeature`)

   3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

   4. Push to the branch (`git push origin feature/AmazingFeature`)

   5. Open a pull request

---

## 📝 License

   This repository is licensed under the Apache License 2.0.
   See the [LICENSE](./LICENSE) file for more details.
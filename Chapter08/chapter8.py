# ***************************************************************************************************************************************
#                                                                                                                                       *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 8                                               *
#                                                                                                                                       *
#       This code demonstrates a comprehensive workflow for loading, training, and evaluating a GraphSAGE model using PyTorch           *
#       Geometric on the PPI (Protein-Protein Interaction) dataset. It begins by setting a fixed random seed for reproducibility,       *
#       then defines a PPIDataLoader class for managing the loading and preparation of training, validation, and test datasets.         *
#       This class uses neighbor sampling to create data loaders tailored for the PPI dataset. The GraphSAGETrainer class is used       *
#       to initialize and manage the GraphSAGE model, including its training and evaluation. The model is trained over a specified      *
#       number of epochs, with training and validation metrics reported periodically. After training, the model's performance is        *
#       evaluated on the test set, with the final F1-score providing an assessment of its accuracy. The code integrates these           *
#       components to ensure a structured approach to handling graph data, training models, and evaluating performance.                 *
#                                                                                                                                       *
# ***************************************************************************************************************************************



# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import SAGEConv, GraphSAGE as PyGGraphSAGE
    

# ---------------------------------------------------------------- Set random seeds for reproducibility ----------------------------------------------------------------

class RandomSeedSetup:
    def __init__(self, seed=0):
        self.seed = seed
        self.setup()

    def setup(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
# ------------------------------------------------- class is designed to handle loading and managing graph datasets ----------------------------------------------------

class GraphDataset:
    def __init__(self, dataset_name):
        self.dataset = Planetoid(root='.', name=dataset_name)
        self.data = self.dataset[0]

    def print_dataset_info(self):
        print(f'Dataset: {self.dataset}')
        print('-------------------')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of nodes: {self.data.x.shape[0]}')
        print(f'Number of features: {self.dataset.num_features}')
        print(f'Number of classes: {self.dataset.num_classes}')

        print(f'\nGraph:')
        print('------')
        print(f'Training nodes: {sum(self.data.train_mask).item()}')
        print(f'Evaluation nodes: {sum(self.data.val_mask).item()}')
        print(f'Test nodes: {sum(self.data.test_mask).item()}')
        print(f'Edges are directed: {self.data.is_directed()}')
        print(f'Graph has isolated nodes: {self.data.has_isolated_nodes()}')
        print(f'Graph has loops: {self.data.has_self_loops()}')

    def create_loader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=[5, 10],
            batch_size=16,
            input_nodes=self.data.train_mask,
        )
    
# -------------------------------------------------- class is used for visualizing subgraphs from a data loader --------------------------------------------------------

class GraphVisualizer:
    def __init__(self, train_loader):
        self.train_loader = train_loader

    def plot_subgraphs(self):
        fig = plt.figure(figsize=(16,16))
        for idx, (subdata, pos) in enumerate(zip(self.train_loader, [221, 222, 223, 224])):
            G = to_networkx(subdata, to_undirected=True)
            ax = fig.add_subplot(pos)
            ax.set_title(f'Subgraph {idx}', fontsize=24)
            plt.axis('off')
            nx.draw_networkx(G,
                            pos=nx.spring_layout(G, seed=0),
                            with_labels=False,
                            node_color=subdata.y,
                            )
        plt.show()
    
#  -------------------------------------- class defines a GraphSAGE model using the SAGEConv layers from PyTorch Geometric -------------------------------------------

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return h

    def fit(self, loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(epochs+1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            for batch in loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss.item()
                acc += self.accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += self.accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
            if epoch % 20 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {loss/len(loader):.3f} | Train Acc: {acc/len(loader)*100:>6.2f}% | Val Loss: {val_loss/len(loader):.2f} | Val Acc: {val_acc/len(loader)*100:.2f}%')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = self.accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

    def accuracy(self, pred_y, y):
        return ((pred_y == y).sum() / len(y)).item()
    
# ----------------------------------------------- class is responsible for loading and preparing the PPI datasets ----------------------------------------------------

class PPIDataLoader:
    def __init__(self, batch_size=2048):
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_datasets(self):
        self.train_dataset = PPI(root=".", split='train')
        self.val_dataset = PPI(root=".", split='val')
        self.test_dataset = PPI(root=".", split='test')

    def create_loaders(self):
        train_data = Batch.from_data_list(self.train_dataset)
        self.train_loader = NeighborLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_neighbors=[20, 10],
            num_workers=2,
            persistent_workers=True
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2)
    
# ---------------------------------------------- class handles the training and evaluation of the GraphSAGE model ----------------------------------------------------

class GraphSAGETrainer:
    def __init__(self, train_loader, val_loader, test_loader, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None

    def setup_model(self, in_channels, hidden_channels, out_channels):
        self.model = PyGGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=out_channels
        ).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

    def evaluate(self):
        test_f1 = self.test(self.test_loader)
        print(f'Test F1-score: {test_f1:.4f}')
    
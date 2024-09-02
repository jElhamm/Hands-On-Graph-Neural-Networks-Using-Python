# *******************************************************************************************************************************************
#                                                                                                                                           *
#                               Hands On Graph Neural Networks Using Python  -  CHAPTER 10                                                  *
#                                                                                                                                           *
#       This script sets up and runs experiments with two graph neural network models, VGAE and DGCNN, using the Cora dataset from          *
#       PyTorch Geometric. It first defines several classes: Encoder for the VGAE encoder network, VGAEModel for the VGAE model             *
#       including training and testing methods, SealProcessing for processing graph data with special transformations, and DGCNNModel       *
#       for a deep graph convolutional neural network with 1D convolutional layers. The GraphModeling class orchestrates data loading,      *
#       model training, and evaluation for both models. It initializes the dataset, processes it, and trains the VGAE model to predict      *
#       graph link probabilities, evaluating performance with metrics like AUC and average precision. It also trains the DGCNN model        *
#       for edge classification, with separate loaders for positive and negative edges.                                                     *
#       Finally, it prints out the results from both models' evaluations.                                                                   *
#                                                                                                                                           *
# *******************************************************************************************************************************************


# !pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.sparse.csgraph import shortest_path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, VGAE, global_sort_pool
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, BCEWithLogitsLoss
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------- Set random seed ----------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(0)
    
# -------------------------------- implements the encoder part of the Variational Graph Autoencoder (VGAE) model --------------------------------------


class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
# ------------------------------- manages the VGAE model, including its initialization, training, and evaluation --------------------------------------

class VGAEModel:
    def __init__(self, dataset, device):
        self.device = device
        self.model = VGAE(Encoder(dataset.num_features, 16)).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, train_data):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(train_data.x, train_data.edge_index)
        loss = self.model.recon_loss(z, train_data.pos_edge_label_index) + \
               (1 / train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    
# ---------------------------------------- This class contains a static method for processing graph data ----------------------------------------------

class SealProcessing:
    @staticmethod
    def process(dataset, edge_label_index, y):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph([src, dst], 2, dataset.edge_index, relabel_nodes=True)
            src, dst = mapping.tolist()
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]
            src, dst = (dst, src) if src > dst else (src, dst)
            adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()
            idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
            adj_wo_src = adj[idx, :][:, idx]
            idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
            adj_wo_dst = adj[idx, :][:, idx]
        
            d_src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
            d_src = np.insert(d_src, dst, 0, axis=0)
            d_src = torch.from_numpy(d_src)
            d_dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
            d_dst = np.insert(d_dst, src, 0, axis=0)
            d_dst = torch.from_numpy(d_dst)
            dist = d_src + d_dst
            z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
            z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
            z = z.to(torch.long)
            node_labels = F.one_hot(z, num_classes=200).to(torch.float)
            node_emb = dataset.x[sub_nodes]
            node_x = torch.cat([node_emb, node_labels], dim=1)
            data = Data(x=node_x, z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)
            
        return data_list
    
# ---------------------------------- This class implements the Deep Graph Convolutional Neural Network (DGCNN) ----------------------------------------

class DGCNNModel(torch.nn.Module):
    def __init__(self, dim_in, k=30):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, 32)
        self.gcn2 = GCNConv(32, 32)
        self.gcn3 = GCNConv(32, 32)
        self.gcn4 = GCNConv(32, 1)
        self.k = k
        self.conv1 = Conv1d(1, 16, 97, 97)
        self.conv2 = Conv1d(16, 32, 5, 1)
        self.maxpool = MaxPool1d(2, 2)
        self.linear1 = Linear(352, 128)
        self.dropout = Dropout(0.5)
        self.linear2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        h1 = self.gcn1(x, edge_index).tanh()
        h2 = self.gcn2(h1, edge_index).tanh()
        h3 = self.gcn3(h2, edge_index).tanh()
        h4 = self.gcn4(h3, edge_index).tanh()
        h = torch.cat([h1, h2, h3, h4], dim=-1)
        h = global_sort_pool(h, batch, k=self.k)
        h = h.view(h.size(0), 1, h.size(-1))
        h = self.conv1(h).relu()
        h = self.maxpool(h)
        h = self.conv2(h).relu()
        h = h.view(h.size(0), -1)
        h = self.linear1(h).relu()
        h = self.dropout(h)
        h = self.linear2(h).sigmoid()
        return h
    
# ------------------------------------------------ This class is the main driver for the experiments -------------------------------------------------
    
class GraphModeling:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Planetoid('.', name='Cora', transform=RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True))
        self.train_data, self.val_data, self.test_data = self.dataset[0]

    def run_vgae(self):
        vgae_model = VGAEModel(self.dataset, self.device)
        for epoch in range(301):
            loss = vgae_model.train(self.train_data)
            val_auc, val_ap = vgae_model.test(self.test_data)
            if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')
        test_auc, test_ap = vgae_model.test(self.test_data)
        print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')
        z = vgae_model.model.encode(self.test_data.x, self.test_data.edge_index)
        Ahat = torch.sigmoid(z @ z.T)
        print(Ahat)
    
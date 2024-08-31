import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GINConv,GraphSAGE,global_max_pool, global_mean_pool, global_min_pool

class GCN(nn.Module):
    def __init__(self, num_features, num_classes, pooling='mean'):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = nn.Linear(64, num_classes)
        self.pooling = pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'min':
            x = global_min_pool(x, batch)
        else:
            raise ValueError('Invalid pooling type.')
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, alpha):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        self.attentions = nn.ModuleList(
            [GATConv(in_features, out_features, heads=1, concat=True, dropout=dropout, alpha=alpha) for _ in
             range(num_heads)])

    def forward(self, g, feature):
        out = feature
        for i, attn in enumerate(self.attentions):
            out = attn(g, out)
            if i != 0:
                out = F.dropout(out, self.dropout, training=self.training)
        return F.elu(out)


class GAT(nn.Module):
    def __init__(self, num_features, num_classes, num_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        self.conv1 = GATLayer(num_features, 64, num_heads=num_heads, dropout=dropout, alpha=alpha)
        self.conv2 = GATLayer(64, num_classes, num_heads=num_heads, dropout=dropout, alpha=alpha)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)

    class GraphSAGELayer(nn.Module):
        def __init__(self, in_features, out_features, num_samples):
            super(GraphSAGELayer, self).__init__()
            self.num_samples = num_samples
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, g, feature):
            out = feature
            for i in range(self.num_samples):
                neighbor_features = dgl.sampling.sample_neighbors(g, g.ndata['id'], num_neighbors=self.num_samples)
                neighbor_features = neighbor_features.ndata['h']
                out = self.linear(torch.cat((out, neighbor_features), dim=1))
            return F.relu(out)

    class GraphSAGE(nn.Module):
        def __init__(self, num_features, num_classes, num_samples):
            super(GraphSAGE, self).__init__()
            self.num_samples = num_samples
            self.conv1 = GraphSAGEConv(num_features, 64, num_samples=num_samples)
            self.conv2 = GraphSAGEConv(64, num_classes, num_samples=num_samples)

        def forward(self, g, features):
            x = self.conv1(g, features)
            x = F.relu(x)
            x = self.conv2(g, x)
            return F.log_softmax(x, dim=1)

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super(GINLayer, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_layers)])

    def forward(self, g, feature):
        out = feature
        for i, layer in enumerate(self.linear):
            out = layer(out)
            if i != 0:
                out = F.relu(out)
        return out

class GIN(nn.Module):
    def __init__(self, num_features, num_classes, num_layers):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(GINLayer(num_features, 64, num_layers=num_layers), aggregator_type='mean')
        self.conv2 = GINConv(GINLayer(64, num_classes, num_layers=num_layers), aggregator_type='mean')

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)






















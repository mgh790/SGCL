import copy
import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose
from src.utils import  feature_drop_weights, drop_feature_weighted_2,  pseudo_drop_weights,cal_Weights
class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)
class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def other_transform(args,data,device, predict_lbl_pro, degree_sim,  weights, Z=None):
        drop_feature_rate_1=args.drop_feature_rate_1
        drop_edge_rate_1=args.drop_edge_rate_1
        node_w_ps,new_index, new_prediction=cal_Weights(Z, predict_lbl_pro, degree_sim, device, weights)
        node_w_ps=node_w_ps.to(device)
        drop_weights = pseudo_drop_weights(data, node_w_ps).to(device)
        node_pseduo = degree_sim.float().to(device)
        feature_weights = feature_drop_weights(data.x, node_c=node_pseduo).to(device)
        edge_index1 = drop_edge_weighted(data.edge_index, drop_weights, drop_edge_rate_1, threshold=0.7)

        x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
        return edge_index1, x_1,new_index, new_prediction


import os
import os.path as osp
import random
import numpy as np
from torch_geometric.utils import  to_undirected,to_networkx
import torch

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{}_{}_".format(name, val)
        st += st_

    return st[:-1]

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def compute_accuracy(preds, labels, train_mask, val_mask, test_mask):

    train_preds = preds[train_mask]
    val_preds = preds[val_mask]
    test_preds = preds[test_mask]

    train_acc = (torch.sum(train_preds == labels[train_mask])).float() / ((labels[train_mask].shape[0]))
    val_acc = (torch.sum(val_preds == labels[val_mask])).float() / ((labels[val_mask].shape[0]))
    test_acc = (torch.sum(test_preds == labels[test_mask])).float() / ((labels[test_mask].shape[0]))
    
    train_acc = train_acc * 100
    val_acc = val_acc * 100
    test_acc = test_acc * 100

    return train_acc, val_acc, test_acc


def masking(fold, data, label_rate=0.01):
        
    # pubmed
    if label_rate == 0.03:
        train_mask = data.train_mask0_03[fold] ; val_mask = data.val_mask0_03[fold] ; test_mask = data.test_mask0_03[fold]
    elif label_rate == 0.06:
        train_mask = data.train_mask0_06[fold] ; val_mask = data.val_mask0_06[fold] ; test_mask = data.test_mask0_06[fold]
    elif label_rate == 0.1:
        train_mask = data.train_mask0_1[fold] ; val_mask = data.val_mask0_1[fold] ; test_mask = data.test_mask0_1[fold]
    
    # Amazon
    elif label_rate == 0.15:
        train_mask = data.train_mask0_15[fold] ; val_mask = data.val_mask0_15[fold] ; test_mask = data.test_mask0_15[fold]
    elif label_rate == 0.2:
        train_mask = data.train_mask0_2[fold] ; val_mask = data.val_mask0_2[fold] ; test_mask = data.test_mask0_2[fold]
    elif label_rate == 0.25:
        train_mask = data.train_mask0_25[fold] ; val_mask = data.val_mask0_25[fold] ; test_mask = data.test_mask0_25[fold]
    
    # Cora, Citeseer
    elif label_rate == 0.5:
        train_mask = data.train_mask0_5[fold] ; val_mask = data.val_mask0_5[fold] ; test_mask = data.test_mask0_5[fold]
    elif label_rate == 1:
        train_mask = data.train_mask1[fold] ; val_mask = data.val_mask1[fold] ; test_mask = data.test_mask1[fold]
    elif label_rate == 2:
        train_mask = data.train_mask2[fold] ; val_mask = data.val_mask2[fold] ; test_mask = data.test_mask2[fold]

    return train_mask, val_mask, test_mask

def compute_representation(net, data, device):

    net.eval()
    reps = []

    data = data.to(device)
    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0)

    return reps

def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s
def feature_drop_weights_pseudo(x, edge_index):
    # x = x.to(torch.bool).to(torch.float32)
    edge_index_ = to_undirected(edge_index)
    input1=x[edge_index_[0]]
    input2=x[edge_index_[1]]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    w = cos(input1, input2)
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def pseudo_drop_weights(data,s):
    s = s.log()
    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col
    return (s.max() - s) / (s.max() - s.mean())


def similarity(edge_index, node_deg, data):
    graph = to_networkx(data, to_undirected=True)
    cos_similarity = torch.nn.CosineSimilarity(dim=0)
    num_nodes = data.num_nodes
    sim = torch.zeros((num_nodes, num_nodes))
    a = torch.zeros((num_nodes, num_nodes))
    degree_sim = torch.zeros(num_nodes)
    predict_lbl = torch.zeros(num_nodes, dtype=torch.long)
    neighbors_dict = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        index, neighborIdx = edge_index[0, i], edge_index[1, i]
        input1, input2 = data.x[index], data.x[neighborIdx]
        similarity_score = cos_similarity(input1, input2)
        sim[index, neighborIdx] = similarity_score
        neighbors_dict[index.item()].append(neighborIdx.item())
        # if similarity_score > 0.02 + 1 / node_deg[neighborIdx]:
        if similarity_score > 0.02 + 1 / node_deg[index]:
            neighbors1 = set(graph.neighbors(int(index)))
            neighbors2 = set(graph.neighbors(int(neighborIdx)))
            common_neighbors = len(neighbors1.intersection(neighbors2)) / (len(neighbors1) * len(neighbors2))
            a[index, neighborIdx] = node_deg[index]+common_neighbors
    for index, neighbors in neighbors_dict.items():
        if len(neighbors)!=0:
            degree_sim[index] = (a[index, :].sum())/len(neighbors)+0.001
            # degree_sim[index] = a[index, :].sum()
    return degree_sim, predict_lbl
def cal_Weights(out, predict_lbl_pro, degree_sim,device, weights):
    predict_lbl_pro=predict_lbl_pro.to(device)
    degree_sim=degree_sim.to(device)
    equal_mask = predict_lbl_pro.int() == torch.argmax(out, dim=1)
    weights = torch.where(equal_mask, torch.ones_like(weights), degree_sim)
    # weights = torch.where(equal_mask, torch.ones_like(weights), z_normalize.mean(dim=1) + degree_sim)
    return weights, 0,0

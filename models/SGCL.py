from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected
from embedder import embedder
from src.sampling import Sampler
from src.transform import other_transform
from src.utils import reset, set_random_seeds, masking,similarity

class SGCL_Trainer(embedder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def _init_model(self):
        self.model = SGCL(self.encoder, self.classifier, self.unique_labels,
                           self.args.tau,
                           self.args.thres,
                           self.args.device).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        # self.adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1])
        self.weights = node_deg / max(node_deg)
        self.nclass = self.data.y.max().item() + 1
    def _init_dataset(self):
        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1])
        self.degree_sim, self.predict_lbl_pro = similarity(edge_index_, node_deg, self.data)

    def train(self):
        for fold in range(self.args.folds):
            set_random_seeds(fold)
            self.train_mask, self.val_mask, self.test_mask = masking(fold, self.data, self.args.label_rate)
            self._init_dataset()
            self.Sampler = Sampler(self.args, self.data, self.labels, self.running_train_mask)
            self._init_model()
            epo = []
            for epoch in range(1, self.args.epochs + 1):
                self.model.train()
                self.optimizer.zero_grad()
                # anchor  = self.transform1(self.data)
                positive=self.transform2(self.data)
                label_matrix, support_index, self.batch_size = self.Sampler.sample()
                pos_rep = self.model(positive)
                pos_support_rep = pos_rep[support_index]
                sup_loss = 0.
                logits, _ = self.model.cls(positive)
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                # sup_loss += F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
                out2 = logits
                edge_index1, x_1, new_index, new_prediction = other_transform(self.args, self.data, self.device,
                                                                              self.predict_lbl_pro, self.degree_sim,
                                                                              self.weights,out2)
                # anchor = Data(x=self.data.x, edge_index=edge_index1)
                anchor = Data(x=x_1, edge_index=edge_index1)
                anchor_rep = self.model(anchor)
                anchor_support_rep = anchor_rep[support_index]
                logits, _ = self.model.cls(anchor)
                # sup_loss += F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                sup_loss /= 2
                out = logits

                ## Pseudo-Labeling
                loss_pseudo = F.cross_entropy(out, torch.argmax(out, dim=1), size_average=True)
                loss_pseudo2 = F.cross_entropy(out, torch.argmax(out2, dim=1), size_average=True)
                lPS = (loss_pseudo + loss_pseudo2) / 2
                ## Supervised Contrastive Learning
                LSupConstrative = self.model.loss(anchor_rep, pos_rep, anchor_support_rep,
                                                   pos_support_rep,
                                                   label_matrix, self.data.y, self.train_mask)
                # Node Similarity Regularization
                LNS = 2 - 2 * F.cosine_similarity(anchor_rep, pos_rep, dim=-1).mean()
                loss = sup_loss + self.args.lam * LSupConstrative + self.args.lam2 * LNS + self.args.lamPseduo * lPS
                st = '[Fold : {}][Epoch {}/{}] Consistency_Loss: {:.4f} | Sup_loss : {:.4f} | Unsup_loss : {:.4f} | lPS : {:.4f} |Total_loss : {:.4f}'.format(
                    fold + 1, epoch, self.args.epochs, LSupConstrative.item(), sup_loss.item(), LNS.item(),
                    lPS.item(),
                    loss.item())

                loss.backward()
                self.loss = loss
                self.optimizer.step()

                epo.append(epoch)
                # evaluation
                self.evaluate(self.data, st)

                if self.cnt == self.args.patience:
                    print("early stopping!")
                    break


            self.optimizer.step()
            self.save_results(fold)
        self.summary()


class SGCL(nn.Module):
    def __init__(self, encoder, classifier, unique_labels, tau=0.1, thres=0.9, device=0):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.tau = tau
        self.thres = thres

        self.softmax = nn.Softmax(dim=1)
        self.num_unique_labels = len(unique_labels)

        self.device = device

        self.reset_parameters()

    def forward(self, x):
        rep = self.encoder(x)
        return rep

    def snn(self, query, supports, labels):
        query = F.normalize(query)
        supports = F.normalize(supports)

        return self.softmax(query @ supports.T / self.tau) @ labels

    def loss(self, anchor, pos, anchor_supports, pos_supports, labels, gt_labels, train_mask):
        with torch.no_grad():
            gt_labels = gt_labels[train_mask].unsqueeze(-1)
            matrix = torch.zeros(train_mask.sum().item(), self.num_unique_labels).to(self.device)
            gt_matrix = matrix.scatter_(1, gt_labels, 1)

        probs1 = self.snn(anchor, anchor_supports, labels)
        with torch.no_grad():
            targets1 = self.snn(pos, pos_supports, labels)
            values, _ = targets1.max(dim=1)
            boolean = torch.logical_or(values > self.thres, train_mask)
            indices1 = torch.arange(len(targets1))[boolean]
            targets1[targets1 < 1e-4] *= 0
            targets1[train_mask] = gt_matrix
            targets1 = targets1[indices1]
        probs1 = probs1[indices1]
        loss = torch.mean(torch.sum(torch.log(probs1 ** (-targets1)), dim=1))
        return loss  # , probs1, targets1

    def cls(self, x):
        out = self.encoder(x)
        return self.classifier(out)


    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)

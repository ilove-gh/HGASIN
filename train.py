from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score

from models import HGDGC
from utils import set_env, Metrics
from process import load_dataset, cross_validation, CustomDataset
import uuid


parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.025,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001791,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=63,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.25, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.4, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--layers', type=int, default=64,
                    help='Number layers: At least greater than 2 to take effect, GCN at least two layers.')
parser.add_argument('--kfold', type=int, default=5,
                    help='k-fold: Cross-validate the fold.')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch_size: Each time the batch_size is read.')
parser.add_argument('--patience', type=int, default=64, help='Patience')
parser.add_argument('--dataset', type=str, default='multimodal-NC_AB-with-MCI',
                    help='Dataset string')
parser.add_argument('--diffusion', type=str, default='Heat',
                    help='diffusion procession')
# multimodal-NC_AB-with-MCI
parser.add_argument('--name', type=str, default='ALL',
                    help='Dataset string')

args = parser.parse_args()
print('The list of model initialization parameters is {}'.format(args.__dict__))

checkpt_file = './pretrained/' + uuid.uuid4().hex + '.pt'

set_env(seed=args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

path = os.getcwd() + '/' + args.dataset
adj, features, labels = load_dataset(dataset=path, balence_weights=False, normalized=None, symbol=args.name)

# Shuffle the data randomly
shuffle_idx = np.random.permutation(len(labels))
labels = labels[shuffle_idx]
adj = adj[shuffle_idx]
features = features[shuffle_idx]

# For cross-validation, get the labels of the training and test sets for all folds
train_index_list, test_index_list = cross_validation(labels, k_fold=args.kfold)


t_total = time.time()
metrics = Metrics()
for fold_index in range(args.kfold):
    # Model and optimizer
    model = HGDGC(nfeat=features.shape[2],
                nlayers=args.layers,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda=args.lamda,
                alpha=args.alpha,
                variant=args.variant,
                is_enhanced=True,
                method=args.diffusion,
                device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training set index
    fold_train_index = train_index_list[fold_index]
    # Validation set index
    fold_test_index = test_index_list[fold_index]

    train_dataset = CustomDataset(adj[fold_train_index], features[fold_train_index], labels[fold_train_index])
    test_dataset = CustomDataset(adj[fold_test_index], features[fold_test_index], labels[fold_test_index])

    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    _pred, _label = [], []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (adj_batch, features_batch, labels_batch) in enumerate(train_dataset):
            adj_batch, features_batch, labels_batch = adj_batch.numpy(), features_batch.numpy(), labels_batch.to(device)
            output = model(features_batch, adj_batch)
            loss = criterion(output, labels_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

            pred = torch.softmax(output, dim=1)
            pred = torch.max(pred, 1)[1].view(-1)
            _pred += pred.detach().cpu().numpy().tolist()
            _label += labels_batch.cpu().numpy().tolist()

        # 求所有batch size的平均损失
        epoch_loss /= (batch_idx + 1)
        acc = accuracy_score(_label, _pred)
        print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, acc))

    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for batch_idx, (adj_batch, features_batch, labels_batch) in enumerate(test_dataset):
            adj_batch, features_batch, labels_batch = adj_batch.numpy(), features_batch.numpy(), labels_batch.to(device)
            pred = torch.softmax(model(features_batch, adj_batch), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += labels_batch.cpu().numpy().tolist()
        metrics.calculate_all_metrics(test_label, test_pred, average='weighted')

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('The accuracy is {}.'.format(metrics.accuracy_in_all_metrics()))
print('The list of model initialization parameters is {}'.format(args.__dict__))
print(
    'The average accuracy is {}%.'.format(metrics.accuracy_average_in_all_metrics()))
metrics.average_all_metrics()








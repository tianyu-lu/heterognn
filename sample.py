import argparse
import os
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import HeteroConv, Linear, GENConv
from torch_geometric.loader import DataLoader

from transformMD import LigandMPNNTransform

import numpy as np



class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('residue', 'near', 'residue'): GENConv((-1, -1), hidden_channels, edge_dim=5*5*16, expansion=4, aggr='add'),
                ('residue', 'near', 'atom'): GENConv((-1, -1), hidden_channels, edge_dim=5*16, expansion=4, aggr='add'),
                ('atom', 'near', 'residue'): GENConv((-1, -1), hidden_channels, edge_dim=5*16, expansion=4, aggr='add'),
                ('atom', 'near', 'atom'): GENConv((-1, -1), hidden_channels, edge_dim=16, expansion=4, aggr='add'),
            }, aggr='sum')
            self.convs.append(conv)

        self.bn = nn.BatchNorm1d(hidden_channels)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        for conv in self.convs:
            x_dict = conv(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            x_dict = {key: self.bn(x) for key, x in x_dict.items()}
        return self.lin(x_dict['residue'])



class UserPDB(Dataset):

    def __init__(self, transform, user_pdb_fp):

        self.transform = transform
        self.user_pdb_fp = user_pdb_fp

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):

        try:
            item = self.transform(user_pdb_fp=self.user_pdb_fp)

        except Exception as err:
            print(f"EXCEPTION: {err}")
            print(f"========================{self.user_pdb_fp} failed =======================")

        return item



@torch.no_grad()
def test(model, loader, local_rank):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(f'cuda:{local_rank}')

        out = model(data)
        loss = F.cross_entropy(out, data['residue'].y, reduction='sum')
        y_true.extend(data['residue'].y.tolist())
        pred = out.argmax(dim=-1)
        y_pred.extend(pred.tolist())
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    acc = np.mean(y_pred == y_true)

    total_loss = torch.tensor([np.sqrt(loss_all / total)]).to(f'cuda:{local_rank}')
    
    return total_loss, acc, y_true, y_pred, out


def runner(args, log_dir, local_rank, rep=None):

    test_dataset = UserPDB(transform=LigandMPNNTransform(), user_pdb_fp=args.user_pdb_fp)

    test_loader = DataLoader(test_dataset, args.batch_size)

    model = HeteroGNN(hidden_channels=args.hidden_dim, out_channels=output_dim, num_layers=args.num_layers)
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    model = model.to(f'cuda:{local_rank}')
    
    test_file = os.path.join(log_dir, f'LigandMPNN-rep{rep}.pt')
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, acc, y_true_test, y_pred_test, out = test(model, test_loader, local_rank)
    print(f'\tTest Loss {test_loss}, Test Accuracy {acc}')
    torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)

    return y_true_test, y_pred_test, out


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_pdb_fp', type=str, default='2V1B.pdb')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ckpt', type=str, default="heterognn_weights.pt")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rep', type=int, default=0)
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up the environement variable for the distributed mode
    local_rank = '0'

    # return an object representing the device on which tensors will be allocated.
    device = torch.device(device)

    # enable benchmark mode in cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path(f"LigandMPNN_Results_{now}")
    log_dir.mkdir(exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    y_true_test, y_pred_test, out = runner(args, device, log_dir, local_rank, args.rep, test_mode=True)

    print(y_true_test.shape)
    print(y_pred_test.shape)
    print(out.shape)

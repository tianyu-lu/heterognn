import pickle
import argparse
import os
import time
import datetime
import h5py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import HeteroConv, Linear, GENConv
from torch_geometric.loader import DataLoader

import utils
from transformMD import LigandMPNNTransform
from graph import atoms_residue_map

import numpy as np



class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network for protein design given ligand atomic context
    """
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



class LigandMPNNDataset(Dataset):
    """
    Code adapted from MISATO ProtDataset class
    """

    def __init__(self, md_data_file, idx_file, transform, qm_data_file=None):
        """
        Args:
            md_data_file (str): H5 file path
            idx_file (str): path of txt file which contains pdb ids for a specific split such as train, val or test.
            transform (obj): class that convert a dict to a PyTorch Geometric graph.
            qm_data_file (str): path to QM data (per-atom numerical features from quantum mechanics calculations)
        """

        self.md_data_file = Path(md_data_file).absolute()

        with open(idx_file, 'r') as f: 
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.md_data_file, 'r') 

        self.transform = transform

        self.qm_data_file = qm_data_file
        if self.qm_data_file is not None:
            self.qm_data_file = Path(qm_data_file).absolute()

            self.qm_f = h5py.File(self.qm_data_file, 'r')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if not 0 <= (index) < len(self.ids):
            raise IndexError(index)

        frame = 0 # NOTE: for now, train on the first frame only

        cache_fp = Path(f"cache/{self.ids[index]}_{frame}.pkl")
        if Path(cache_fp).exists():
            with open(cache_fp, "rb") as cfp:
                item = pickle.load(cfp)
        else:
            passing = False
            while not passing:
                item = self.f[self.ids[index]]
                qm_item = self.qm_f[self.ids[index]] if self.qm_data_file is not None else None

                try:
                    item = self.transform(item, frame=frame, qm_item=qm_item)
                    ### convert to PDB then use RFdiffusion PDB parser
                    # item = self.transform(item, frame=frame, pdb=self.ids[index], qm_item=qm_item)
                    passing = True

                    with open(cache_fp, "wb") as cfp:
                        pickle.dump(item, cfp)

                except Exception as err:
                    print(f"EXCEPTION: {err}")
                    print(f"========================{self.ids[index]} failed =======================")
                    index = torch.randint(len(self.ids), (1,)).item() # randomly sample a new item
                    pass
    
        return item



def train_loop(model, loader, optimizer, local_rank):
    """One epoch of training

    Parameters
    ----------
    model : HeteroGNN
        Heterogeneous GNN model instance
    loader : 
        PyG dataloader
    optimizer
        Torch optimizer
    local_rank
        GPU rank

    Returns
    -------
    Tuple of [Total training loss, Validation accuracy]
    """
    model.train()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(f'cuda:{local_rank}')

        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data['residue'].y, reduction='sum')
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()

        y_true.extend(data['residue'].y.tolist())
        pred = out.argmax(dim=-1)
        y_pred.extend(pred.tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    acc = np.mean(y_pred == y_true)

    total_loss = torch.tensor([np.sqrt(loss_all / total)]).to(f'cuda:{local_rank}')

    return total_loss, acc


@torch.no_grad()
def test(model, loader, local_rank):
    """Evaluate on test data

    Parameters
    ----------
    model : HeteroGNN
        Heterogeneous GNN model instance
    loader : 
        PyG dataloader
    local_rank
        GPU rank

    Returns
    -------
    Tuple of [Test loss, test accuracy, true labels, predicted labels]
    """
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
    
    return total_loss, acc, y_true, y_pred


def train(args, device, log_dir, local_rank, rep=None, test_mode=False):
    """Main function to train the model

    Parameters
    ----------
    args : User-specified arguments
    device : torch.device()
    log_dir : Directory to log outputs
    local_rank : GPU number
    rep : repetition number, optional
    test_mode : bool, optional
        Evaluate the model if True, otherwise train the model, by default False

    Returns
    -------
    Tuple of [Best validation loss, best validation accuracy]
    """

    qm_data_file = args.qm_file if args.qm else None
    train_dataset = LigandMPNNDataset(args.mdh5_file, idx_file=args.train_set, transform=LigandMPNNTransform(noise=args.noise, K=args.knn_res, M=args.knn_atom), qm_data_file=qm_data_file)
    val_dataset = LigandMPNNDataset(args.mdh5_file, idx_file=args.val_set, transform=LigandMPNNTransform(noise=args.noise, K=args.knn_res, M=args.knn_atom), qm_data_file=qm_data_file)
    test_dataset = LigandMPNNDataset(args.mdh5_file, idx_file=args.test_set, transform=LigandMPNNTransform(noise=args.noise, K=args.knn_res, M=args.knn_atom), qm_data_file=qm_data_file)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)

    num_workers = 8
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=num_workers, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=num_workers, sampler=test_sampler)

    for data in train_loader:
        num_features = data.num_features
        break

    model = HeteroGNN(hidden_channels=args.hidden_dim, out_channels=len(atoms_residue_map), num_layers=args.num_layers)
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    model = model.to(f'cuda:{local_rank}')
    
    model_without_ddp = model

    best_val_loss = 999
    best_acc = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss, train_acc = train_loop(model, train_loader, optimizer, local_rank)
        val_loss, acc, y_true, y_pred = test(model, val_loader, local_rank)
        val_loss = val_loss.item()
        if utils.is_main_process():
            if val_loss < best_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss.item(),
                    }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                best_val_loss = val_loss
                best_acc = acc
        elapsed = (time.time() - start)
        if utils.is_main_process():
            print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
            print('\tTrain Loss: {:.7f}, Train Accuracy: {:.7f}, Val Loss: {:.7f}, Val Accuracy: {:.7f}'.format(train_loss.item(), train_acc, val_loss, acc))
            if epoch == 1:
                print("Number of Parameters")
                print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if test_mode:
        train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        if utils.is_main_process():
            model.load_state_dict(cpt['model_state_dict'])
        _, _, y_true_train, y_pred_train = test(model, train_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, y_true_val, y_pred_val = test(model, val_loader, local_rank)
        if utils.is_main_process():
            torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        test_loss, acc, y_true_test, y_pred_test = test(model, test_loader, local_rank)
        if utils.is_main_process():
            print(f'\tTest Loss {test_loss}, Test Accuracy {acc}')
            torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)

    return best_val_loss, best_acc


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdh5_file', type=str, default="../data/MD/h5_files/MD.hdf5")
    parser.add_argument('--train_set', type=str, default="../data/MD/splits/train_MD.txt")
    parser.add_argument('--val_set', type=str, default="../data/MD/splits/val_MD.txt")
    parser.add_argument('--test_set', type=str, default="../data/MD/splits/test_MD.txt")
    # if overfitting on small dataset:
    # parser.add_argument('--mdh5_file', type=str, default="../data/MD/h5_files/tiny_md.hdf5")
    # parser.add_argument('--train_set', type=str, default="../data/MD/splits/train_tinyMD.txt")
    # parser.add_argument('--val_set', type=str, default="../data/MD/splits/val_tinyMD.txt")
    # parser.add_argument('--test_set', type=str, default="../data/MD/splits/test_tinyMD.txt")
    parser.add_argument('--qm_file', type=str, default="../data/QM/h5_files/QM.hdf5")
    parser.add_argument('--qm', action='store_true')
    parser.add_argument( '--master_port', type=int, default=12354)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=32) # 16
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=128) # 64
    parser.add_argument('--num_layers', type=int, default=2) # 2
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--knn_res', type=int, default=48) # 48
    parser.add_argument('--knn_atom', type=int, default=25) # 25
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    # set up the environement variable for the distributed mode
    local_rank = '0'

    # return an object representing the device on which tensors will be allocated.
    device = torch.device(device)

    # enable benchmark mode in cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', f"{now}_K{args.knn_res}_M{args.knn_atom}")
        else:
            log_dir = os.path.join('logs', log_dir)
        if utils.is_main_process():
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        train(args, device, log_dir, local_rank)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', f'MD_{now}')
            print("log_dir", log_dir)
            if utils.is_main_process():
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, local_rank, rep, test_mode=True)


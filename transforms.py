'''
Code adapted from:

MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''


import torch
from torch_geometric.data import HeteroData
from graph import ligandmpnn_to_graph


def prot_ligand_graph_transform(item, K, M, noise, frame, pdb=None, qm_item=None, user_pdb_fp=None):
    """Organize node, edges, and their features into a PyG HeteroData object

    Parameters
    ----------
    item : str
        Name of PDB
    K : int
        Number of nearest neighbors where a protein residue node has edges with other protein residue nodes
    M : int
        Number of nearest neighbors with which a protein residue node has edges with ligand atom nodes
    noise : float
        IID Gaussian noise to add to coordinates to perturb input data
    frame : int
        Index of frame of trajectory to featurize
    pdb : str, optional
        If not None, uses the PDB parser instead of the h5 parser, by default None
    qm_item : str, optional
        If not None, uses QM features for ligand atoms, by default None
    user_pdb_fp : str, optional
        If not None, uses user input PDB path as input, by default None

    Returns
    -------
    HeteroData
        Pytorch Geometric hetero-data object
    """

    res_node_feats,res_node_labels,atom_node_feats,res_res_edge_index,res_atom_edge_index,atom_atom_edge_index,res_res_edge_feats,res_atom_edge_feats,atom_atom_edge_feats = ligandmpnn_to_graph(item, K, M, noise, frame, pdb=pdb, qm_item=qm_item, user_pdb_fp=user_pdb_fp)

    data = HeteroData()

    data['residue'].x = res_node_feats # [num_residues, num_features_residue]
    data['residue'].y = res_node_labels # [num_residues, num_restypes]
    data['atom'].x = atom_node_feats # [num_atoms, num_features_atom]

    data['residue', 'near', 'residue'].edge_index = res_res_edge_index # [2, num_edges_residue_residue] (2, L*K)
    data['residue', 'near', 'atom'].edge_index = res_atom_edge_index # [2, num_edges_residue_atom] (2, L*M)
    data['atom', 'near', 'residue'].edge_index = torch.cat((res_atom_edge_index[1:2], res_atom_edge_index[0:1]), dim=0) # [2, num_edges_residue_atom] (2, L*M)
    data['atom', 'near', 'atom'].edge_index = atom_atom_edge_index # [2, num_edges_atom_atom] (2, L*M*M)

    data['residue', 'near', 'residue'].edge_attr = res_res_edge_feats # [num_edges_residue_residue, num_features_residue_residue]
    data['residue', 'near', 'atom'].edge_attr = res_atom_edge_feats # [num_edges_residue_atom, num_features_residue_atom]
    data['atom', 'near', 'residue'].edge_attr = res_atom_edge_feats # [num_edges_residue_atom, num_features_residue_atom]
    data['atom', 'near', 'atom'].edge_attr = atom_atom_edge_feats # [num_edges_atom_atom, num_features_atom_atom]
    return data

'''MISATO, a database for protein-ligand interactions
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

from collections import defaultdict
import copy
import itertools
import pickle
import subprocess
import h5py
import numpy as np
import scipy.spatial as ss
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
# from torch_sparse import coalesce

num2atom = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'CL', 'BR', 'I', 'B', 'NA', 'MG', 'AL', 'SI', 'K', 'CA', 'SE']
atom2num = {x:i for i,x in enumerate(num2atom)}

# constants from RFdiffusion
num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]  

aa2num = {x:i for i,x in enumerate(num2aa)}
aa2num["CYX"] = aa2num["CYS"]
aa2num["HIE"] = aa2num["HIS"]

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
]

atom_mapping = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F', 5:'P', 6:'S', 7:'CL', 8:'BR', 9:'I', 10: 'UNK'}
residue_mapping = {0:'ALA', 1:'ARG', 2:'ASN', 3:'ASP', 4:'CYS', 5:'CYX', 6:'GLN', 7:'GLU', 8:'GLY', 9:'HIE', 10:'ILE', 11:'LEU', 12:'LYS', 13:'MET', 14:'PHE', 15:'PRO', 16:'SER', 17:'THR', 18:'TRP', 19:'TYR', 20:'VAL', 21:'UNK'}

ligand_atoms_mapping = {8: 0, 16: 1, 6: 2, 7: 3, 1: 4, 15: 5, 17: 6, 9: 7, 53: 8, 35: 9, 5: 10, 33: 11, 26: 12, 14: 13, 34: 14, 44: 15, 12: 16, 23: 17, 77: 18, 27: 19, 52: 20, 30: 21, 4: 22, 45: 23}

# atoms_residue gives residue type
# atoms_type gives type according to MISATO nomenclature
# atoms_number gives atomic number
atoms_residue_map = {0: 'MOL', 1: 'ACE', 2: 'ALA', 3: 'ARG', 4: 'ASN',
                         5: 'ASP', 6: 'CYS',7: 'CYX', 8: 'GLN', 9: 'GLU',
                         10: 'GLY', 11: 'HIE', 12: 'ILE', 13: 'LEU', 14: 'LYS',
                         15: 'MET', 16: 'PHE', 17: 'PRO', 18: 'SER', 19: 'THR',
                         20: 'TRP', 21: 'TYR', 22: 'VAL'}

atomic_numbers_map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',
                      12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',
                      19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}


with open('atoms_type_map.pickle', 'rb') as f1:
  atoms_type_map = pickle.load(f1)

with open('atoms_name_map_for_pdb.pickle', 'rb') as f2:
  pdb_atom_map = pickle.load(f2)

# fixes 😱
pdb_atom_map[('GLU', 14, 'O2')] = "OE2"
pdb_atom_map[('ASP', 11, 'O2')] = "OD2"

def one_of_k_encoding_unk_indices(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    one_hot_encoding = [0] * len(allowable_set)
    if x in allowable_set:
        one_hot_encoding[x] = 1
    else:
        one_hot_encoding[-1] = 1
    return one_hot_encoding

def one_of_k_encoding_unk_indices_qm(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    one_hot_encoding = [0] * (len(allowable_set)+1)
    if x in allowable_set:
        one_hot_encoding[allowable_set[x]] = 1
    else:
        one_hot_encoding[-1] = 1
    return one_hot_encoding


'''
<KeysViewHDF5 ['atoms_element', 'atoms_number', 'atoms_residue',
               'atoms_type', 'frames_bSASA', 'frames_distance',
               'frames_interaction_energy', 'frames_rmsd_ligand',
               'molecules_begin_atom_index', 'trajectory_coordinates']>
'''

class Protein:

  """
  bb_coords: (B, N, 4, 3)
  ligand_coords: [(B, N, 4, 3)]
  frame: int or list (frame in trajectory)
  ligand_res_edges: List[List[Tuple]] (ligand_idx, residue_idx) both start at zero
  """
  def __init__(self, item, noise=0.1, frame=None, lig_res_dist=10, res_res_dist=10, pdb=None, user_pdb_fp=None):
    self.pdb=pdb
    self.item = item
    self.noise = noise
    self.lig_res_dist = lig_res_dist
    self.res_res_dist = res_res_dist
    self.user_pdb_fp = user_pdb_fp
    
    if self.user_pdb_fp is not None:
      self.bb_coords, self.seq, self.ligand_coords, self.ligand_atoms = self.direct_extract_info_pdb(self.user_pdb_fp)
    if self.pdb is not None:
      self.bb_coords, self.seq, self.ligand_coords, self.ligand_atoms = self.extract_info_pdb(frame=frame, pdb=pdb)
    else:
      self.bb_coords, self.seq, self.ligand_coords, self.ligand_atoms = self.extract_info(frame=frame)
    if self.ligand_coords is None:
      self.ligand_coords, self.ligand_atoms = self.extract_info_ligand(frame=frame)
    self.bb_coords = self.get_pseudo_cb(self.bb_coords)
    self.bb_coords = self.add_noise(self.bb_coords, noise=noise)
    self.ligand_coords = self.add_noise(self.ligand_coords, noise=noise)

    cb = 4
    self.protein_trees = [KDTree(c[:, cb]) for c in self.bb_coords]
    self.ligand_trees = [KDTree(c) for ligand_coord in self.ligand_coords for c in ligand_coord]

    self.ligand_res_edges = self.get_lig_res_edges()

  def add_noise(self, arr_list, noise=0.1):
    """Add element-wise Gaussian noise with variance = noise"""
    return [arr + np.random.randn(*arr.shape) * noise for arr in arr_list]

  def get_atom_name(self, residue_atom_index, residue_name, type_string):
    """Adapted from MISATO"""
    if residue_name != 'MOL':
      try:
        atom_name = pdb_atom_map[(residue_name, residue_atom_index-1, type_string)]
      except KeyError:
        atom_name = type_string+str(residue_atom_index)
    else:
      atom_name = None
    return atom_name

  def update_residue_indices(self, i, residue_number, residue_atom_index, type_string, residue_name):
    """
    Adapted from MISATO
    If the atom sequence has adjacent O-N atoms, increase the residue_number
    """
    atoms_type = self.item['atoms_type']
    atoms_residue = self.item['atoms_residue']
    if i < len(atoms_type)-1:
      if type_string[0] == 'O' and atoms_type_map[atoms_type[i+1]][0] == 'N' or atoms_residue_map[atoms_residue[i+1]]=='MOL':
        # GLN has a O N sequence within the AA
        if not ((residue_name == 'GLN' and residue_atom_index==12) or (residue_name == 'ASN' and residue_atom_index==9)):
            residue_number +=1
            residue_atom_index = 0
    return residue_number, residue_atom_index

  def translate_to_pdb_atoms(self):
    """
    Adapted from MISATO
    We go through each atom line and bring the inputs in the pdb format
    """
    atoms_type = np.asarray(self.item['atoms_type'])
    atoms = []
    residue_number = 1
    residue_atom_index = 0
    for i, a in enumerate(atoms_type):
        residue_atom_index +=1
        type_string = atoms_type_map[a]
        residue_name = atoms_residue_map[self.item['atoms_residue'][i]] # three letter aa name
        try:
            atom_name = self.get_atom_name(residue_atom_index, residue_name, type_string)
        except:
            print("PDB Map Error")
            # import ipdb; ipdb.set_trace()
        atoms.append(atom_name)
        residue_number, residue_atom_index = self.update_residue_indices(i, residue_number, residue_atom_index,
                                                                         type_string, residue_name)
    return atoms

  def get_ligand_idx(self):
    molecule_idx = np.asarray(self.item['molecules_begin_atom_index'])
    residues = np.asarray(self.item['atoms_residue'])
    n_atoms = len(residues)
    possible_idx = molecule_idx.tolist()
    possible_idx.append(n_atoms)
    final_idx = []
    MOL = 0
    for i in range(len(molecule_idx)):
      start, end = possible_idx[i], possible_idx[i+1]
      if np.all(residues[start:end] == MOL):
        final_idx.append((start, end))
    return final_idx

  def extract_info(self, frame=None):
    c = np.asarray(self.item['trajectory_coordinates'])
    atom_numbers = np.asarray(self.item['atoms_number'])
    if frame is not None:
      c = np.expand_dims(c[frame], 0)

    # protein information
    pdb_atom_translation = np.asarray(self.translate_to_pdb_atoms())

    bb_atoms = [['N'], ['CA'], ['C'], ['O', 'OXT']]
    for i in range(len(pdb_atom_translation)-1):
        if pdb_atom_translation[i] == 'O' and pdb_atom_translation[i+1] == 'OXT':
            pdb_atom_translation[i] = 'OE1'
    extract_idx = lambda atom: np.concatenate(np.argwhere(np.isin(pdb_atom_translation, atom)))
    bb_atom_idx = [extract_idx(atom) for atom in bb_atoms]

    try:
        bb_coords = np.stack([c[:, idx] for idx in bb_atom_idx], axis=2)
    except:
        print(f"Extract Error: {[len(a) for a in bb_atom_idx]}")
        # import ipdb; ipdb.set_trace()

    seq = np.asarray(self.item['atoms_residue'])[bb_atom_idx[0]]
    # seq = [atoms_residue_map[res] for res in seq] # NOTE changed

    # ligand information
    ligand_idx = self.get_ligand_idx()
    ligand_coords = [c[:, start:end] for start, end in ligand_idx]
    convert = lambda atoms: [atomic_numbers_map[n] for n in atoms.tolist()]
    # ligand_atoms = [convert(atom_numbers[start:end]) for start, end in ligand_idx] # NOTE changed
    ligand_atoms = [atom_numbers[start:end] for start, end in ligand_idx]

    # print("bb_coords")
    # print(bb_coords, bb_coords.shape)
    # print("seq")
    # print(seq, seq.shape)
    # print("ligand_coords")
    # print(ligand_coords, ligand_coords[0].shape)
    # print("ligand_atoms")
    # print(ligand_atoms, ligand_atoms[0].shape)
    
    # bb_coords: (1, num_residues, 4, 3) 4 = 'N', 'CA', 'C', 'O'
    # seq: (num_residues,)
    # ligand_coords: [(1, num_atoms, 3)]
    # ligand_atoms: [(num_atoms,)]
    return bb_coords, seq, ligand_coords, ligand_atoms
  
  def extract_info_ligand(self, frame=None):
    c = np.asarray(self.item['trajectory_coordinates'])
    atom_numbers = np.asarray(self.item['atoms_number'])
    if frame is not None:
      c = np.expand_dims(c[frame], 0)

    # ligand information
    ligand_idx = self.get_ligand_idx()
    ligand_coords = [c[:, start:end] for start, end in ligand_idx]
    convert = lambda atoms: [atomic_numbers_map[n] for n in atoms.tolist()]
    # ligand_atoms = [convert(atom_numbers[start:end]) for start, end in ligand_idx] # NOTE changed
    ligand_atoms = [atom_numbers[start:end] for start, end in ligand_idx]

    return ligand_coords, ligand_atoms

  # PDB parsers adapted from RFdiffusion

  def parse_pdb_lines(self, lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    # pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num
    # NOTE: remove .strip() to handle empty chain ID
    pdb_idx = [( l[21:22], int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num
    
    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        try:
          idx = pdb_idx.index((chain,resNo))
        except ValueError:
          # print(f"Missing CA for residue {resNo}")
          continue
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break
        
    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0 
    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)
    
    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }
    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            # if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'): # NOTE: changed
            if l[17:20] == "MOL" and not (ignore_het_h and l[77]=='H'):
                info_het.append(atom2num[l[76:78].strip().upper()])
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = np.array(info_het)

    return out

  def parse_pdb(self, filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return self.parse_pdb_lines(lines, **kwargs)

  def extract_info_pdb(self, frame=None, pdb=None):
    assert frame is not None
    assert pdb is not None
    subprocess.run([
      "python3", "/sdf/group/cryoem/g/CS57/mpnn/misato_dataset/src/data/processing/h5_to_pdb.py",
      "--struct", pdb,
      "--frame", str(frame),
      "--datasetMD", "/sdf/group/cryoem/g/CS57/mpnn/misato_dataset/data/MD/h5_files/MD.hdf5",
      "--mapdir", "/sdf/group/cryoem/g/CS57/mpnn/misato_dataset/src/data/processing/Maps/"
    ], check=True)
    pdb_fp = f"/sdf/group/cryoem/g/CS57/mpnn/misato_dataset/examples/cache/{pdb}_MD_frame{frame}.pdb"
    out = self.parse_pdb(pdb_fp, parse_hetatom=True, ignore_het_h=False)

    # TODO: QM features concat to info_het?
    # print("================================xyz_het", out["xyz_het"].shape)
    bb_coords = out["xyz"][None, :, :4, :]
    seq = out["seq"]
    try:
      ligand_coords = [out["xyz_het"][None, :, :]]
      ligand_atoms = [out["info_het"]]
    except IndexError: # give up and get coords from h5 file
      ligand_coords = None
      ligand_atoms = None

    # bb_coords: (1, num_residues, 4, 3) 4 = 'N', 'CA', 'C', 'O'
    # seq: (num_residues,)
    # ligand_coords: [(1, num_atoms, 3)]
    # ligand_atoms: [(num_atoms,)]
    return bb_coords, seq, ligand_coords, ligand_atoms
  
  def direct_extract_info_pdb(self, pdb_fp):
    
    out = self.parse_pdb(pdb_fp, parse_hetatom=True, ignore_het_h=False)

    bb_coords = out["xyz"][None, :, :4, :]
    seq = out["seq"]
    try:
      ligand_coords = [out["xyz_het"][None, :, :]]
      ligand_atoms = [out["info_het"]]
    except IndexError as err: # give up and get coords from h5 file
      print(f"Parsing PDB had error {err}")

    # bb_coords: (1, num_residues, 4, 3) 4 = 'N', 'CA', 'C', 'O'
    # seq: (num_residues,)
    # ligand_coords: [(1, num_atoms, 3)]
    # ligand_atoms: [(num_atoms,)]
    return bb_coords, seq, ligand_coords, ligand_atoms


  # c dimensions: b n 5 3
  def get_pseudo_cb(self, coord):
    N = coord[:, :, 0]
    Ca = coord[:, :, 1]
    C = coord[:, :, 2]
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c, axis=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    return np.concatenate([coord, Cb[..., None, :]], axis=2)

  # returns list of dicts pairing ligand atoms to interacting residues
  def get_lig_res_edges(self):
    matches = []
    for ligand_tree, protein_tree in zip(self.ligand_trees, self.protein_trees):
      matches.append(ligand_tree.query_ball_tree(protein_tree, self.lig_res_dist))

    results = []
    for ligands in matches:
      single_ligand_match = []
      for i, sublist in enumerate(ligands):
        if len(sublist) != 0:
          single_ligand_match += [(i, x) for x in sublist]
      results.append(copy.deepcopy(single_ligand_match))
    return results


# from pathlib import Path
# md_fp = Path("/sdf/group/cryoem/g/CS57/mpnn/misato_dataset/data/MD/h5_files/MD.hdf5")

# md_data = h5py.File(md_fp, 'r')

# for k in md_data.keys():
#     print(f"=========================={k}==========================")
#     Protein(md_data[k], frame=0)


def prot_df_to_graph(item, df, edge_dist_cutoff, feat_col='element'):
    r"""
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric, where each node is an atom.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param node_col: Column of dataframe to find node feature values. For example, for atoms use ``feat_col="element"`` and for residues use ``feat_col="resname"``
    :type node_col: str, optional
    :param allowable_feats: List containing all possible values of node type, to be converted into 1-hot node features. 
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :type allowable_feats: list, optional
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two atoms, defaults to 4.5.
    :type edge_dist_cutoff: float, optional

    :return: tuple containing

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.

        - edges (torch.LongTensor): Edges in COO format

        - edge_weights (torch.LongTensor): Edge weights, defined as a function of distance between atoms given by :math:`w_{i,j} = \frac{1}{d(i,j)}`, where :math:`d(i, j)` is the Euclidean distance between node :math:`i` and node :math:`j`.

        - node_pos (torch.FloatTensor): x-y-z coordinates of each node
    :rtype: Tuple
    """ 

    allowable_feats = atom_mapping
    try : 
        node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edges = torch.LongTensor(edge_tuples).t().contiguous()
        edges = to_undirected(edges)
    except:
        print(f"Problem with PDB Id is {item['id']}")
        
    node_feats = torch.FloatTensor([one_of_k_encoding_unk_indices(e-1, allowable_feats) for e in df[feat_col]])
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)

    
    return node_feats, edges, edge_weights, node_pos



def rbf(A, B):
    """Adapted from ProteinMPNN
    A: (num_edges, 3)
    B: (num_edges, 3)
    """
    D = np.sqrt(np.sum((A - B)**2,-1) + 1e-6)
    D_min, D_max, D_count = 2., 22., 16
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = np.expand_dims(D, -1)
    RBF = np.exp(-((D_expand - D_mu) / D_sigma)**2)
    return torch.from_numpy(RBF)


def get_rbf(A, B, edge_index):
    """
    Given two arrays of shape (L, num_A, 3) and (L, num_B, 3),
    along with the edge index (i, j) (shape (K,2))
        where i is the node index of A along the L dimension
        where j is the node index of B along the L dimension
    Computes all pairs of distances (num_A * num_B many)
        and bins them with rbf
    """
    edge_index = edge_index.numpy()
    num_A, num_B = A.shape[-2], B.shape[-2] # the combinatorial dimension, should be 1 for atoms
    i, j = edge_index[0, :], edge_index[1, :]
    all_rbf = [rbf(A[i, ai, :], B[j, bi, :]) for ai, bi in itertools.product(range(num_A), range(num_B))]
    all_rbf = torch.stack(all_rbf).permute(1, 0, 2)
    # combine last two dimensions
    return all_rbf.reshape(-1, all_rbf.shape[-2]*all_rbf.shape[-1]).type(torch.FloatTensor)


def ligandmpnn_to_graph(item, K, M, noise, frame, pdb=None, qm_item=None, user_pdb_fp=None):
    """TODO: Docstring
    """ 

    protein = Protein(item, noise, frame=frame, pdb=pdb, user_pdb_fp=user_pdb_fp)

    res_node_labels = []
    res_res_edge_tuples = []
    res_atom_edge_tuples = []
    atom_atom_edge_tuples = []

    protein_tree = protein.protein_trees[0]
    protein_xyz = protein.bb_coords[0]
    ligand_xyz = protein.ligand_coords[0]

    for resi, ca in enumerate(protein_xyz[:, 1, :]):

        # edge_tuples = list(protein_tree.query_pairs(10.0)) # alternative to RBF

        dists, neighs = protein_tree.query(ca, k=K)
        edge_tuples = [(resi, j) for j in neighs]
        res_res_edge_tuples.extend(edge_tuples)
        res_node_labels.append(protein.seq[resi])


    res_res_edge_index = torch.LongTensor(res_res_edge_tuples).t().contiguous()
    res_res_edge_index = to_undirected(res_res_edge_index)
    # res_res_edge_index = remove_self_loops(res_res_edge_index)[0]
    res_res_edge_feats = get_rbf(protein_xyz, protein_xyz, res_res_edge_index) # [7498, 25, 16]

    res_node_feats = torch.zeros(len(protein_xyz), 23)
    res_node_labels = torch.LongTensor(res_node_labels)

    # get features for res-atom and atom-atom edges
    for ligand_res in protein.ligand_res_edges:
      # group by protein index
      res_ligand_map = defaultdict(list)
      for ligand_idx, res_idx in ligand_res:
          res_ligand_map[res_idx].append(ligand_idx)
      
      for res_idx in res_ligand_map.keys():
          # fully connected graph per protein index
          ligand_idx_neigh = list(set(res_ligand_map[res_idx]))
          atom_atom_edge_tuples.extend([(i,j) for i,j in itertools.product(ligand_idx_neigh, ligand_idx_neigh)])

          # alternative: one ligand graph per protein node
          # atomic_context = torch.FloatTensor([one_of_k_encoding_unk_indices(protein.ligand_atoms[0][i], atom_mapping) for i in ligand_idx_neigh])

          # connect all atoms to their neighboring residue
          res_atom_edge_tuples.extend([(res_idx, j) for j in ligand_idx_neigh])


    res_atom_edge_index = torch.LongTensor(res_atom_edge_tuples).t().contiguous()
    ligand_xyz = ligand_xyz.transpose(1, 0, 2)
    res_atom_edge_feats = get_rbf(protein_xyz, ligand_xyz, res_atom_edge_index) # [872, 5, 16]

    atom_atom_edge_index = torch.LongTensor(atom_atom_edge_tuples).t().contiguous()
    atom_atom_edge_index = coalesce(atom_atom_edge_index)
    atom_atom_edge_feats = get_rbf(ligand_xyz, ligand_xyz, atom_atom_edge_index) # [3443, 1, 16] one graph for ligand atoms

    # one graph for ligand atoms
    atom_node_feats = torch.FloatTensor([one_of_k_encoding_unk_indices(e, atom_mapping) for e in protein.ligand_atoms[0]]) # [num_atoms, 11]

    # concatenate QM features
    if qm_item is not None:
      qm_feats = torch.from_numpy(qm_item["atom_properties"]["atom_properties_values"][()]).to(atom_node_feats)
      atom_node_feats = torch.cat((atom_node_feats, qm_feats), dim=-1)

    # edge_weights = torch.FloatTensor(
    #     [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)

    return (
        res_node_feats,
        res_node_labels,
        atom_node_feats,
        res_res_edge_index,
        res_atom_edge_index,
        atom_atom_edge_index,
        res_res_edge_feats,
        res_atom_edge_feats,
        atom_atom_edge_feats,
    )


def mol_df_to_graph_for_qm(df, bonds=None, allowable_atoms=None, edge_dist_cutoff=4.5, onehot_edges=True):
    """
    Converts molecule in dataframe to a graph compatible with Pytorch-Geometric
    :param df: Molecule structure in dataframe format
    :type mol: pandas.DataFrame
    :param bonds: Molecule structure in dataframe format
    :type bonds: pandas.DataFrame
    :param allowable_atoms: List containing allowable atom types
    :type allowable_atoms: list[str], optional
    :return: Tuple containing \n
        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by atom type in ``allowable_atoms``.
        - edge_index (torch.LongTensor): Edges from chemical bond graph in COO format.
        - edge_feats (torch.FloatTensor): Edge features given by bond type. Single = 1.0, Double = 2.0, Triple = 3.0, Aromatic = 1.5.
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node.
    """
    if allowable_atoms is None:
        allowable_atoms = ligand_atoms_mapping
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
    
    if bonds is not None:
        N = df.shape[0]
        bond_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
        bond_data = torch.FloatTensor(bonds)
        edge_tuples = torch.cat((bond_data[:, :2], torch.flip(bond_data[:, :2], dims=(1,))), dim=0)
        edge_index = edge_tuples.t().long().contiguous()
        
        if onehot_edges:
            bond_idx = list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist())) + list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist()))
            edge_attr = F.one_hot(torch.tensor(bond_idx), num_classes=4).to(torch.float)
            # edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
  
        else:
            edge_attr = torch.cat((torch.FloatTensor(bond_data[:,-1]).view(-1), torch.FloatTensor(bond_data[:,-1]).view(-1)), dim=0)
    else:
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edge_index = torch.LongTensor(edge_tuples).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_attr = torch.FloatTensor([1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edge_index.t()]).view(-1)
        edge_attr = edge_attr.unsqueeze(1)
    
    node_feats = torch.FloatTensor([one_of_k_encoding_unk_indices_qm(e, allowable_atoms) for e in df['element']])    
    
    return node_feats, edge_index, edge_attr, node_pos

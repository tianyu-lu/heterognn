'''
Code adapted from 

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

from transforms import prot_ligand_graph_transform


class LigandMPNNTransform(object):
    """
    Transform backbone coordinates and ligand atoms into a heterogeneous graph
    """

    def __init__(self, K=48, M=25, noise=0.1):
        """

        Args:
            K (float, optional): K nearest neighbors to construct residue-residue edges
            M (float, optional): M nearest neighbors per residue to construct residue-atom edges
        """
        self.K = K
        self.M = M
        self.noise = noise

    def __call__(self, item=None, frame=0, pdb=None, qm_item=None, user_pdb_fp=None):
        
        item = prot_ligand_graph_transform(item, K=self.K, M=self.M, noise=self.noise, frame=frame, pdb=pdb, qm_item=qm_item, user_pdb_fp=user_pdb_fp)
        return item

    

# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
import numpy as np
import os

#pip install parmed

# only get the protein part of the trajectory
def extract_pro(psf, xtc):
    # determine the output file path
    dirname = os.path.dirname(xtc)
    basename = os.path.basename(xtc)

    # extract
    u = mda.Universe(psf, xtc)
    protein = u.select_atoms('protein')
    protein_psf = protein.convert_to("PARMED")
    out_psf = os.path.join(dirname, basename.split('.')[0] + '_protein.psf')
    out_xtc = os.path.join(dirname, basename.split('.')[0] + '_protein.xtc')
    if os.path.isfile(out_xtc) == False:
        protein_psf.save(out_psf)
        with mda.Writer(out_xtc, protein.n_atoms) as W:
            for ts in u.trajectory:
                W.write(protein)
    # return absolute path
    return out_psf, out_xtc

# protein centering and alignment
def traj_align(psf, xtc, out_xtc, center=True):
    """
    psf: PSF; TPR for the simulation system;
    xtc: XTC or DCD format;
    out_xtc: aligned output trajectory path;
    center: whether center the protein to the center of the box;

    !!! default: align the trajectory with respect to the first frame [hard coded!]
    return None
    """ 
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    
    # Center protein in the center of the box
    if center:
        protein = u.select_atoms('protein')
        not_protein = u.select_atoms('not protein')
        for ts in u.trajectory:
            protein.unwrap(compound='fragments')
            protein_center = protein.center_of_mass(pbc=True)
            dim = ts.triclinic_dimensions
            box_center = np.sum(dim, axis=0) / 2
            # translate all atoms
            u.atoms.translate(box_center - protein_center)
            # wrap all solvent part back to the box
            not_protein.wrap(compound='residues')

    # align using C-alpha atoms
    align.AlignTraj(u, # universe object; trajectory to align
                    ref, # reference
                    select='name CA', # selection of atoms to align
                    filename=out_xtc,
                    match_atoms=True,
                   ).run()

    return None

def traj_align_onfly(psf, xtc, out_xtc, center=True):
    """
    psf: PSF; TPR for the simulation system;
    xtc: XTC or DCD format;
    out_xtc: aligned output trajectory path;
    center: whether center the protein to the center of the box;

    !!! default: align the trajectory with respect to the first frame [hard coded!]
    return None
    """
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    if center:
        protein = u.select_atoms('protein')
        not_protein = u.select_atoms('not protein')
        transforms = [trans.unwrap(protein),
                trans.center_in_box(protein, wrap=True),
                trans.wrap(not_protein)]

        u.trajectory.add_transformations(*transforms)
    
    # align using C-alpha atoms
    align.AlignTraj(u, # universe object; trajectory to align
                    ref, # reference
                    select='name CA', # selection of atoms to align
                    filename=out_xtc,
                    match_atoms=True,
                   ).run()

    return None

def heavy_atom_templete(pdb, new_pdb,new_gro):
    u = mda.Universe(pdb)
    heavy_atoms = u.select_atoms('not name H* and not (resname AMN or resname CBX)')
    heavy_atoms.write(new_pdb)
    heavy_atoms.write(new_gro)
    return None    

def CA_atom_templete(pdb, new_pdb,new_gro):
    u = mda.Universe(pdb)
    CA_atoms = u.select_atoms('name CA')
    CA_atoms.write(new_pdb)
    CA_atoms.write(new_gro)
    return None

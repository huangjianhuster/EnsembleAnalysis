# Author: Jian Huang
# Date: Oct 9, 2023
# email: jianhuang@umass.edu

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis.analysis.rms as rms
import mdtraj as md

# RMSD calculation
def traj_rmsd(psf, xtc, align_select, rmsd_list):
    """
    psf: PSF or TPR
    xtc: already_aligned XTC or DCD
    return rmsd_matrix
        rmsd_matrix has a shape of (4, number_of_frames)
        rmsd_matrix[0]: frames
        rmsd_matrix[1]: time
        rmsd_matrix[2]: rmsd of align_select
        rmsd_matrix[3-]: rmsd of rmsd_list
    """
    u = mda.Universe(psf, xtc)
    ref = mda.Universe(psf, xtc)
    ref.trajectory[0]
    R = mda.analysis.rms.RMSD(u,ref,select=align_select,groupselections=rmsd_list)
    R.run()
    return R.results.rmsd.T

# RMSF calculation: only for C-alpha
def traj_rmsf(psf, xtc):
    """
    psf: PSF or TPR
    xtc: already_aligned XTC or DCD
    return resid, rmsf_matrix
    """
    u = mda.Universe(psf, xtc)
    # ref = mda.Universe(psf, xtc)
    # ref.trajectory[0]
    c_alphas = u.select_atoms('protein and name CA')
    R = rms.RMSF(c_alphas).run()
    rmsf_matrix = R.results.rmsf
    return c_alphas.resnums, rmsf_matrix

# radius of gyration (IDP protein)
def traj_rg(psf, xtc):
    """
    psf: PSF or TPR
    xtc: XTC or DCD
    return Rgyr
    """
    u = mda.Universe(psf, xtc)
    Rgyr = []
    for ts in u.trajectory:
        Rgyr.append( u.atoms.radius_of_gyration())
    Rgyr = np.array(Rgyr)
    return Rgyr

# helicity & sheet
def traj_ss(psf, xtc):
    if xtc.endswith("xtc"):
        traj = md.load(xtc, top=psf)
    if xtc.endswith("dcd"):
        traj = md.load_dcd(xtc, top=psf)
#    residues = list(traj.topology.residues)
    dssp = md.compute_dssp(traj, simplified=True)
    helicity = np.where(dssp=='H', 1, 0)
    helicity_ave = np.sum(helicity, 0) / helicity.shape[0]
    sheet = np.where(dssp=='E', 1, 0)
    sheet_ave = np.sum(sheet, 0) / sheet.shape[0]
    return  helicity_ave, sheet_ave

# End-to-end distance
def endtoend(psf,dcd):
    u = mda.Universe(psf,dcd)  # always start with a Universe
    # can access structure via segid (s4AKE) and atom name
    nterm = u.select_atoms('protein and name N')[0]
    # ... takes the last atom named 'C'
    cterm = u.select_atoms('protein and name C')[-1]
    bb = u.select_atoms('protein and backbone')
    ete = []
    frame = []
    for ts in u.trajectory:  # iterate through all frames
        r = cterm.position - nterm.position  # end-to-end vector from atom positions
        d = numpy.linalg.norm(r)   # end-to-end distance
        frame.append(ts.frame)
        ete.append(d)
    return frame, ete

# PCA analysis
def PCA(psf,dcd):
    aligner = align.AlignTraj(u, u, select='backbone', in_memory=True).run()
    pc = pca.PCA(u, select='backbone', align=True, mean=None, n_components=None).run()
    backbone = u.select_atoms('backbone')
    n_bb = len(backbone)
    return pc

# write pdbs into trajectory <-- for VAE decoder-generated trajectory
def pdbs2xtc(pdblist, out_xtc):
    """
    pdblist: list of pdb paths to be combined together into a single trajectory file
    out_xtc: output trajectory path
    """
    # to determin PDB atoms
    u = mda.Universe(pdblist[0])
    with mda.Writer(out_xtc, len(u.atoms)) as xtc_writer:
        for pdb in pdblist:
            u.load_new(pdb)
            xtc_writer.write(u)

    return None

# get bonds
def get_bonds(psf, xtc, top, atom1_name, atom2_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; ; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    every: frequency of calculation
    use_atom_type: whether use atom type; default=True
    return: bond length array; unit: Angstrom
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_bonds = []
    for ts in u.trajectory[::int(every)]:
        bond_list = []
        for bond in protein_selection.bonds:

            # conditions to select atom names for bonds
            if not use_atom_type:

                atom1 = bond.atoms[0].name
                atom2 = bond.atoms[1].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    bond_list.append(bond.value())
            else:
                atom1 = bond.atoms[0].type
                atom2 = bond.atoms[1].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    bond_list.append(bond.value())

        all_bonds.append(bond_list)
    gmx_top = pmd.load_file(top)
    bond_len = next((bond for bond in topology.bonds if bond.atom1.name == atom1_name \
                    and bond.atom2.name == atom2_name), None)
    b0 = equilibrium_length = bond_N_O.type.req
    return np.array(all_bonds), b0

# get angles
def get_angles(psf, xtc, top, atom1_name, atom2_name, atom3_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    atom3_name: atomname of the third atom
    every: frequency of calculation
    use_atom_type: whether use atom type; default=False
    return: angle list in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_angles = []
    for ts in u.trajectory[::int(every)]:
        angle_list = []
        for angle in protein_selection.angles:

            # conditions to select atom names for bonds
            if not use_atom_type:

                atom1 = angle.atoms[0].name
                atom2 = angle.atoms[1].name
                atom3 =  angle.atoms[2].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    angle_list.append(angle.value())
            else:
                atom1 = angle.atoms[0].type
                atom2 = angle.atoms[1].type
                atom3 = angle.atoms[2].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    angle_list.append(angle.value())

        all_angles.append(angle_list)
    gmx_top = pmd.load_file(top)
    angle_N_CA_CB = next((angle for angle in gmx_top.angles if angle.atom1.name == atom1_name \
                        and angle.atom2.name == atom2_name and angle.atom3.name == atom3_name), None)
    theta0 = equilibrium_angle = angle_N_CA_CB.type.theteq
    return np.array(all_angles), theta0

# dihedrals
def get_dihedrals(psf, xtc, top, atom1_name, atom2_name, atom3_name, atom4_name, every=1, use_atom_type=True):
    """
    psf: PSF
    xtc: XTC
    atom1_name: atomname of the first atom; ; use atomtype if use_atom_type=True
    atom2_name: atomname of the second atom
    atom3_name: atomname of the third atom
    atom4_name: atomname of the fourth atom
    every: frequency of querying the trajectory
    use_atom_type: whether use atom type; default=True
    return: dihedral array in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_dihedrals = []
    for ts in u.trajectory[::int(every)]:
        dihedral_list = []
        for dihedral in protein_selection.dihedrals:

            # conditions to select atom names for bonds
            if not use_atom_type:

                atom1 = dihedral.atoms[0].name
                atom2 = dihedral.atoms[1].name
                atom3 =  dihedral.atoms[2].name
                atom4 = dihedral.atoms[3].name
                # print(atom1, atom2)
                if atom1 == atom1_name and atom2 == atom2_name:
                    dihedral_list.append(dihedral.value())
            else:
                atom1 = dihedral.atoms[0].type
                atom2 = dihedral.atoms[1].type
                atom3 = dihedral.atoms[2].type
                atom4 = dihedral.atoms[3].type
                if atom1 == atom1_name and atom2 == atom2_name:
                    dihedral_list.append(dihedral.value())

        all_dihedrals.append(dihedral_list)
    gmx_top = pmd.load_file(top)
    dihedral_N_CA_C_N = next((dihedral for dihedral in gmx_top.dihedrals if dihedral.atom1.name == atom1_name \
                            and dihedral.atom2.name == atom2_name \
                            and dihedral.atom3.name == atom3_name \
                            and dihedral.atom4.name == atom4_name), None)
    d0 = dihedral_N_CA_C_N.psi_k
    return np.array(all_dihedrals), d0

def get_bb_impropers(psf, xtc, top, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: impropers in degrees
    """
    u = mda.Universe(psf, xtc)
    protein_selection = u.select_atoms("protein")

    all_bbimpropers = []
    for ts in u.trajectory[::int(every)]:
        bbimproper_list = []
        for bbimproper in protein_selection.impropers:
            atom1 = bbimproper.atoms[0].type
            atom2 = bbimproper.atoms[1].type
            atom3 =  bbimproper.atoms[2].type
            atom4 = bbimproper.atoms[3].type
            # print(atom1, atom2)
            if atom1 == 'C' and atom2 == 'CT1' and atom3 == 'NH1' and atom4 == 'O':
                bbimproper_list.append(bbimproper.value())
            elif atom1 == 'NH1' and atom2 == 'C' and atom3 == 'CT1' and atom4 == 'H':
                bbimproper_list.append(bbimproper.value())
        all_bbimpropers.append(bbimproper_list)

    gmx_top = pmd.load_file(top)
    dihedral_imp1 = next((dihedral for dihedral in gmx_top.dihedrals if dihedral.atom1.name == 'C' \
                        and dihedral.atom2.name == 'CT1' and dihedral.atom3.name == 'NH1' \
                        and dihedral.atom4.name == 'O'), None)
    dihedral_imp2 = next((dihedral for dihedral in gmx_top.dihedrals if dihedral.atom1.name == 'NH1' \
                        and dihedral.atom2.name == 'C' and dihedral.atom3.name == 'CT1' \
                        and dihedral.atom4.name == 'H'), None)
    equilibrium_improper1 = improper_dihedral_imp1.psi_k
    equilibrium_improper2 = improper_dihedral_imp2.psi_k

    return np.array(all_bbimpropers), [equilibrium_improper1,equilibrium_improper2]

# get peptide plane omegas
def get_omegas(psf, xtc, top, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: array of omega values (in degrees)
    """
    u = mda.Universe(psf, xtc)
    ags = [res.omega_selection() for res in u.residues[:-1]] # ignore the last residues
    R = Dihedral(ags).run(step=int(every))

    gmx_top = pmd.load_file(top)
    omega = next((dihedral for dihedral in topology.dihedrals if (dihedral.atom1.name == 'C' \
                        and dihedral.atom2.name == 'N' and dihedral.atom3.name == 'CA' \
                        and dihedral.atom4.name == 'C')), None)
    omega0 = omega_dihedral_C_N_CA_C.type.phi_k
    return R.results['angles'], omega0

# get backbone psi and phi angles
def get_phis_psis(psf, xtc, top, every=1):
    """
    psf: PSF
    xtc: XTC
    every: frequency of calculation
    return: array of phi values (in degrees), array of psi values (in degrees)
    """
    u = mda.Universe(psf, xtc)
    ags_phi = [res.omega_selection() for res in u.residues[1:]] # the first residue has no phi
    ags_psi = [res.psi_selection() for res in u.residues[:-1]]  # the last residue has no psi

    R_phi = Dihedral(ags_phi).run(step=int(every))
    R_psi = Dihedral(ags_psi).run(step=int(every))

    gmx_top = pmd.load_file(top)
    phi_dihedral_N_CA_C_N = next((dihedral for dihedral in topology.dihedrals if (dihedral.atom1.name == 'N' \
                        and dihedral.atom2.name == 'CA' and dihedral.atom3.name == 'C' \
                        and dihedral.atom4.name == 'N')), None)
    phi0 = phi_dihedral_N_CA_C_N.type.phi_k

    psi_dihedral_CA_C_N_CA = next((dihedral for dihedral in topology.dihedrals if (dihedral.atom1.name == 'CA' \
                        and dihedral.atom2.name == 'C' and dihedral.atom3.name == 'N' \
                        and dihedral.atom4.name == 'CA')), None)
    psi0 = psi_dihedral_CA_C_N_CA.type.phi_k
    return R_phi.results['angles'], phi0, R_psi.results['angles'], psi0


def Gaussian_distribution(CV,CVo):
    kb = 1.86188e3
    beta = 1/ (1.380649e-23 * 298)
    expo = beta*0.5*kb*(CV - CVo)
    return np.exp(-expo)

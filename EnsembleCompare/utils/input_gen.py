# Author: Jian Huang & Shrishti
# Date: 2023-08-06
# E-mail: jianhuang@umass.edu

# Dependencies
import MDAnalysis as mda
from sklearn.model_selection import train_test_split
from MDAnalysis.analysis.bat import BAT
from MDAnalysis.analysis import dihedrals
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import itertools

def gen_parms_combinations(**params_dict):
    """
    params_dit: searching grid for each hyperparameter
        it has to contain the following keys:
        BATCH_SIZE: [64,]
        LATENT_DIM: [2, 5, 10]
        NUM_HIDDEN_LAYER: [3, 4, 5]
        EPOCHS: [100, 200, 500]
        RATE: [0.001, 0.01, 0.0001]
    return hyperparams_combinations
        <-- list of hyperparameter dict with BATCH_SIZE LATENT_DIM NUM_HIDDEN_LAYER EPOCHS RATE as the keys
    """
    hyperparams_list = [
                  params_dict['BATCH_SIZE'],
                  params_dict['LATENT_DIM'],
                  params_dict['NUM_HIDDEN_LAYER'],
                  params_dict['EPOCHS'],
                  params_dict['RATE'],
                 ]
    combinations = list(itertools.product(*hyperparams_list))
    # reconfigure combinations to dict
    hyperparams_combinations = []
    for i in combinations:
        dict_tmp = {}
        dict_tmp['BATCH_SIZE'], dict_tmp['LATENT_DIM'], dict_tmp['NUM_HIDDEN_LAYER'], dict_tmp['EPOCHS'], dict_tmp['RATE']= i

        hyperparams_combinations.append(dict_tmp)
    return hyperparams_combinations

def get_ic(psf, xtc):
    u = mda.Universe(psf, xtc)
    selection = u.select_atoms("protein")

    # BAT run
    R = BAT(selection)
    R.run()

    # split EC, IC (bonds, angles and dihedrals)
    Ic = R.results.bat[:,9:]
    Ec = R.results.bat[0,:9]

    # split Ic to get bonds, angles and dihedrals
    bonds = R.results.bat[:, 9:len(u.atoms)-3+9]
    angles = R.results.bat[:, len(u.atoms)-3+9:(len(u.atoms)-3)*2+9]
    dihedrals = R.results.bat[:, (len(u.atoms)-3)*2+9:]
    return Ec, bonds, angles, dihedrals, R

def get_xyz(psf, xtc):
    u = mda.Universe(psf, xtc)
    heavy_atoms = u.select_atoms('not name H*')
    xyz = []
    for ts in u.trajectory:
        xyz.append(heavy_atoms.positions)
    xyz_array = np.array(xyz)
    return xyz_array

def get_cxyz(psf, xtc):
    u = mda.Universe(psf, xtc)
    CA_atoms = u.select_atoms('name CA')
    xyz = []
    for ts in u.trajectory:
        xyz.append(CA_atoms.positions)
    xyz_array = np.array(xyz)
    return xyz_array

def get_contact_map(psf, xtc):
    u = mda.Universe(psf, xtc)
    return None    

# https://userguide.mdanalysis.org/1.1.1/examples/analysis/structure/dihedrals.html
def get_bbtorsion(psf, xtc):
    """
    bb_torsion: 2d arrary with a shape of (n_frames, 2*n_atomgroups)
        <-- atomgroups == resude numbers
        for each row:
            torsions are organized as: res1_phi, res1_psi, res2_phi, res2_psi ...
    """

    u = mda.Universe(psf, xtc)
    protein = u.select_atoms("protein")
    
    phis = [res.phi_selection() for res in protein.residues[1:]]
    psis = [res.psi_selection() for res in protein.residues[:-1]]
    phis_traj = dihedrals.Dihedral(phis).run()
    psis_traj = dihedrals.Dihedral(psis).run()

#    diher = np.concatenate((phis_traj.angles,psis_traj.angles),axis=1)
#    bbtorsion = diher*(np.pi/180)   
#    shape = (phis_traj.shape[0], phis_traj.shape[1] + psis_traj.shape[1])
#    bb_torsion = np.empty(shape)
#    bb_torsion[:, ::2] = phis_traj
#    bb_torsion[:, 1::2] = psis_traj
    return phis_traj.angles, psis_traj.angles

def scaling_spliting_dihedrals(arr, split):
    """
    arr: the input array
    return scaler, x_test, x_train
    """
    scaler = MinMaxScaler()
    di_sin = np.sin(arr)
    di_cos = np.cos(arr)

    maps = np.concatenate((di_sin,di_cos),axis=1)
    scale = scaler.fit_transform(maps)

    # split dataset into testing and training
    x_train, x_test, y_train, y_t = train_test_split(maps_scale, scale, test_size=split, random_state=42)
 #   x_val, x_test, y_val, y_test = train_test_split(x_t, y_t, test_size=0.5, random_state=42)
    #x_test = scale[::4]
    #x_train = np.delete(scale, list(range(0, scale.shape[0], 4)), axis=0)
    return scaler, x_test, x_train

def scaling_spliting_cartesian(mps, split):
    maps = mps.reshape(mps.shape[0], -1)
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(maps)
    # split dataset into testing and training
    x_test, x_train, yt, yt = train_test_split(scale, scale, test_size=split, random_state=42)
#    x_val, x_test, yt ,yt  = train_test_split(x_t, x_t, test_size=0.5, random_state=42)
    return scaler, x_test, x_train

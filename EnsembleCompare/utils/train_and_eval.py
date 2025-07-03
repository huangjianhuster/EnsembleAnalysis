# Dependencies
import pickle
import os, sys, datetime
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import MDAnalysis as mda

from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr

import multiprocessing
from multiprocessing import Process
import subprocess
import concurrent.futures

# from utils.model_2 import *
from utils.plot import *

import warnings
# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def training(**kwargs):
    """
    **kwargs: dict to specify hyperparameters:
        {
            "scaler": scaler,
            "x_test": x_test_array,
            "x_train": x_train_array,

            "BATCH_SIZE": 64,
            "LATENT_DIM": 2,
            "NUM_HIDDEN_LAYER": 4,
            "EPOCHS": 200,
            "RATE": 0.00001,
            "early_stopping": True,
            "seed": 20
        }

    """
    seed = kwargs['seed']
    tf.random.set_seed(seed)
    scaler = kwargs['scaler']
    x_test = kwargs['x_test']
    x_train = kwargs['x_train']
    original_dim = x_train.shape[1]

    BATCH_SIZE = kwargs['BATCH_SIZE']
    LATENT_DIM = kwargs['LATENT_DIM']
    NUM_HIDDEN_LAYER = kwargs['NUM_HIDDEN_LAYER']
    EPOCHS = kwargs['EPOCHS']
    RATE = kwargs['RATE']
    early_stopping = kwargs['early_stopping']
    outtraj_dirname = kwargs['outtraj_dirname']

    # Build VAE MODEL
    encoder, decoder, vae = build_vae(original_dim, LATENT_DIM, NUM_HIDDEN_LAYER, RATE)

    # Train VAE MODEL
    log_dir = str(outtraj_dirname) + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= f"{log_dir}/checkpoint",
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    ) 
    if "early_stopping" == False:
        history = vae.fit(x=x_train, y=x_train,
                shuffle=True,
                epochs=EPOCHS,
                validation_split=0.2, # validation_data=(x_val, x_val),
                verbose=2,
                batch_size = BATCH_SIZE,
                callbacks = [tensorboard_callback, model_checkpoint_callback])
    else:
        # If you want early stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        history = vae.fit(x=x_train,#, y=x_train,
                shuffle=True,
                epochs=EPOCHS,
                validation_split=0.2, # validation_data=(x_val, x_val),
                verbose=2,
                batch_size = BATCH_SIZE,
                callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback])
    formatter = mpl.ticker.EngFormatter()
    all_hype = f"B{BATCH_SIZE}LD{LATENT_DIM}HL{NUM_HIDDEN_LAYER}E{EPOCHS}R{RATE}S{kwargs['split']}"

    vae.save(f"{outtraj_dirname}/{all_hype}/vae")
    encoder.save(f"{outtraj_dirname}/{all_hype}/encoder")
    decoder.save(f"{outtraj_dirname}/{all_hype}/decoder")
    # Plot losses
#    plot_label_clusters(vae, x_train, x_train)
    plot1 = train_test_loss_plot(history, outtraj_dirname, all_hype)
    # TEST 
    encoded = encoder.predict(x_test)#, batch_size=BATCH_SIZE)
    decoded = decoder.predict(encoded[2])
    demap =  scaler.inverse_transform(decoded)     
    x_train = scaler.inverse_transform(x_train)
    x_test = scaler.inverse_transform(x_test)
    # Plot here

    # Evaluation 
    Spearmann, Pearson, pv_spearman, pv_pearson, RMSD_mean, RMSD_std = evaluate(encoded, demap, x_test) 
    print(Spearmann, Pearson, pv_spearman, pv_pearson, RMSD_mean, RMSD_std)

    latent_space_plot(encoded, LATENT_DIM, RMSD_mean, f"{outtraj_dirname}/{all_hype}")

    # Create return dict
    return_dict = {}
    
    return_dict['BATCH_SIZE'] = BATCH_SIZE
    return_dict['LATENT_DIM'] = LATENT_DIM
    return_dict['NUM_HIDDEN_LAYER'] = NUM_HIDDEN_LAYER
    return_dict['EPOCHS'] = EPOCHS
    return_dict['RATE'] = RATE
    return_dict['x_test'] = x_test

    return_dict['Spearmann'] = Spearmann
    return_dict['Pearson'] = Pearson 
    return_dict['val_spearman'] = pv_spearman
    return_dict['val_pearson'] = pv_pearson
    return_dict['RMSD_mean'] = RMSD_mean
    return_dict['RMSD_std'] = RMSD_std
    return_dict['demap'] = demap
    return_dict['outtraj_dirname'] = outtraj_dirname
    return_dict['hyper_together'] = all_hype
    return return_dict

# Sampling from Latent space
#def sample_latent():

def evaluate(encoded, demap, x_test):
    Spearmann, Pearson, pv_spearman, pv_pearson = cal_corr(x_test, encoded[2])
    RMSD_mean, RMSD_std = cal_rmsd(x_test, demap)
    return Spearmann, Pearson, pv_spearman, pv_pearson, RMSD_mean, RMSD_std

# Spearmann correlation and Pearson correlation
def cal_corr(data_original, data_encoded):
    dist_encoded = np.square(euclidean_distances(data_encoded, data_encoded)).flatten()
    dist_original = np.square(euclidean_distances(data_original, data_original)).flatten()
    spearman, pv_spearman = spearmanr(dist_original, dist_encoded)
    pearson, pv_pearson = pearsonr(dist_original, dist_encoded)
    return spearman, pearson, pv_spearman, pv_pearson

# RMSD in A
def cal_rmsd(data_original, data_decoded):
    """
    Calculate RMSD between input and output array
    Input eg. (test data, output data after inverse transformation, 
    return mean RMSD and standard deviation 
    """
    data_decoded = data_decoded
    a = np.sum(np.square(data_decoded - data_original), axis=1)
    rmsd = np.sqrt(a / (data_original.shape[1]))
    return np.mean(rmsd), np.std(rmsd)

def dihedral_demap_to_PDB(Ic_bonds, Ic_angles, Ec, torsion,pdb_temp, outtraj_dirname,R):
    Ic_bonds_angles_ave = np.average(np.concatenate([Ic_bonds, Ic_angles], axis=1), axis=0)
    new_fold = f"{outtraj_dirname}/pdb"
    if os.path.exists(new_fold) == False:
        path = os.path.join(outtraj_dirname,"pdb")
        os.mkdir(path)
    i = 0
    file = []
    for tors in torsion:
        final_array = np.concatenate([Ec, Ic_bonds_angles_ave, tors])
        xyz = R.Cartesian(final_array)
    
        # Flatten the array
        new_xyz = xyz.reshape(-1)
    
        # Write into PDB
        template = extract_template(pdb_temp)
        coorinates = load_coor(new_xyz,template)
        write_file(f"{new_fold}/{i}.pdb",coorinates)
        file.append(f"{i}.pdb")
        i += 1  
    with open(f"{new_fold}/pickle_file.pkl", 'wb') as w:
        pickle.dump(file, w)
    pick_file = f"{new_fold}/pickle_file.pkl"
    return pick_file, new_fold

def dihedraltoFull(pdb, demap, selection,new_fold):
    # load a structure as normal, get first chain
    parser = PDBParser()
    myProtein = parser.get_structure("protein",pdb)
    myChain = myProtein[0]
    # compute bond lengths, angles, dihedral angles
    myChain.atom_to_internal_coordinates(verbose=True)
    pdb_files = []
    if selection == 'phi' or selection == 'omega' :
        phi =  [None for _ in range(len(demap))]
        new_demap = np.concatenate((phi,demap),axis=1)
    elif selection == 'psi':
        psi =  [None for _ in range(len(demap))]
        new_demap = np.concatenate((demap,psi),axis=1)
    for j, aa in enumerate(demap):
        resid = myChain.get_residues()
        for i, r in enumerate(resid):
            r.internal_coord.set_angle("phi", aa[i])
        myChain.internal_to_atom_coordinates()
        # write new conformation with PDBIO
        pdb_files.append(f"{new_fold}/{j}myChain.pdb")
        write_PDB(myProtein, f"{new_fold}/{j}myChain.pdb")
    pick_file = f"{new_fold}/pickle_file.pkl"
    with open(pick_file, 'wb') as w:
        pickle.dump(pdb_files, w)
    return pick_file
        

def cartesian_demap_to_PDB(demap, outtraj_dirname, pdb_temp):
    i = 0
    file = []
    for x in demap:
        template = extract_template(pdb_temp)
        coorinates = load_coor( x,template)
        write_file(f"{outtraj_dirname}/{i}.pdb",coorinates)
        file.append(f"{i}.pdb")
        i += 1
    with open(f"{outtraj_dirname}/pickle_file.pkl", 'wb') as w:
        pickle.dump(file, w)
    pick_file = f"{outtraj_dirname}/pickle_file.pkl"
    return pick_file

def run_command(command):
    subprocess.run(command, shell=True)

# CA to fully-atomic
def CAtoFull(pick_file, outtraj_dirname):
    pick_file1 = []
    pdb_file = pickle.load(open(pick_file,'rb'))

    # CA to full atomic software directory path
    ModRef_path = '/home/shrishti/Documents/Projects/IDP_ensemble/ProDynaVAE/ModRefiner-l'
    os.chdir(ModRef_path)

    k = 0
    # uses 12 cpus in parallel    
    for i in range(0,len(pdb_file[:10]),12):
        # To add mainchain atoms from CA pdbs
        commands = [f'./mcrefinement {outtraj_dirname} {ModRef_path} {pdb_file[j]} {pdb_file[j]} 5' for j in range(k, k+12, 1)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(run_command, commands)
        # To build sidechains atoms from mainchain pdb
        #commands1 = [f'./emrefinement {outtraj_dirname}  {ModRef_path} mc{pdb_file[j]} mc{pdb_file[j]} 100 5' for j in range(k, k+12, 1)]
        #with concurrent.futures.ProcessPoolExecutor() as executor:
        #    executor.map(run_command, commands1)
        # Save names to pickle file
        for j in range(k,k+12,1):
            pick_file1.append(f"{outtraj_dirname}/mc{pdb_file[j]}")
            os.system("rm %s" % f"{outtraj_dirname}/mc{pdb_file[j]}")
        k += 12
    for temp_file in pdb_file:
        os.system("rm %s" % f"{outtraj_dirname}/{temp_file}")
    pick_file1_name = f"{outtraj_dirname}/pickle_file_new.pkl"
    with open(pick_file1_name, 'wb') as w:
        pickle.dump(pick_file1, w)
    return pick_file1_name

def PDB_to_XTC(pickle_file, output_xtc,time_s):
    """
        Mutiple PDB to XTC conveter 
    """
    # Load PDB pickle file
    pdb_file = pickle.load(open(pickle_file,'rb'))
#    reformat_pdbfiles = [(i) for i in pdb_file]

    # from pdbs to xtc
    u = mda.Universe(f"{output_xtc}/{pdb_file[0]}")
    xtc_tmp = f"{output_xtc}/output_tmp.xtc"
    xtc_file = f"{output_xtc}/decoder.xtc"  
    with mda.Writer(xtc_tmp, len(u.atoms)) as xtc_writer:
#        for pdb in reformat_pdbfiles[0:]:
        for pdb in pdb_file[0:]:
            u.load_new(f"{output_xtc}/{pdb}")  # Load each PDB file into the Universe
            xtc_writer.write(u)

    # Add time step in the trajectory
    u2 = mda.Universe(pdb_file[0], xtc_tmp)
    select_atoms = u2.select_atoms("all")
    select_atoms.write(f"{output_xtc}/noH.pdb")
    for ts in u2.trajectory:
        ts.data['dt'] = time_s
        ts.data['time'] = ts.frame*time_s
        ts.data['step'] = ts.frame
    with mda.Writer(xtc_file, u2.atoms.n_atoms) as W:
        for ts in u2.trajectory:
            W.write(u2.atoms)
#    for temp_file in pdb_file:
#        os.system("rm %s" % temp_file)
    os.system("rm %s" % xtc_tmp)
    return None

def CA_XTC(pdb, new_fold, demap): 
    pdbs_fold = f"{new_fold}/pdb"
    if os.path.exists(pdbs_fold) == False:
        path = os.path.join(pdbs_fold)
        os.mkdir(pdbs_fold)
    pickle_file = cartesian_demap_to_PDB(demap, pdbs_fold, pdb)
    #pickle_file = CAtoFull(pickle_file, pdbs_fold)
    PDB_to_XTC(pickle_file, pdbs_fold, 1)
    return None

def Dihedral_XTC(pdb, new_fold, demap, selection):
    pdbs_fold = f"{new_fold}/pdb"
    if os.path.exists(pdbs_fold) == False:
        path = os.path.join(pdbs_fold)
        os.mkdir(pdbs_fold)
    pickle_file = dihedraltoFull(pdb, demap, selection, pdbs_fold)
    PDB_to_XTC(pickle_file, pdbs_fold, 1)
    return None

def demap_to_xtc(psf, pdb, demap, remove_selection, out_xtc):
    if remove_selection in ["phi", "psi", "omega"]:
        Dihedral_XTC(pdb, new_fold, demap, selection)
        return
    
    u = mda.Universe(psf)
    nonH_atoms = u.select_atoms(remove_selection)   # example: "not name H*"

    num_frames = demap.shape[0]
    
    # make sure dir has been made
    if os.path.exists(out_xtc) == False:
        os.mkdir(out_xtc)

    nonH = mda.Merge(nonH_atoms)
    nonH.load_new(demap[0].reshape((len(nonH_atoms), 3)))

    # write out a PDB with removed selection
    u_pdb = mda.Universe(pdb)
    decoder_pdb = u_pdb.select_atoms(remove_selection)
    decoder_pdb.write(f"{out_xtc}/decoder.pdb")

    # if remove_selection == "name CA":
    #    CA_XTC(f"{out_xtc}/noH.pdb", out_xtc, demap)

    #else:
    with mda.Writer(f"{out_xtc}/decoder.xtc", len(nonH.atoms)) as xtc_writer:
        for frame in range(num_frames):
            nonH.load_new(demap[frame].reshape((len(nonH_atoms), 3)))
            nonH.trajectory.ts.data['dt'] = 1
            nonH.trajectory.ts.data['time'] = frame
            nonH.trajectory.ts.data['step'] = frame
            xtc_writer.write(nonH.atoms)

    return None


# Decoded xyz coordinates to Protein
def extract_template(pdb_file):
    """
        Extract the original PDB template except coordinates
        Input: Template PDB
        Output: template without coordinates
    """
    template = []
    file = open(pdb_file).readlines()
    for line in file:
        if line.startswith("ATOM"):
            template.append(line[:31] + " " * 23 + line[54:])    
        elif line.startswith("TER"):
            template.append(line)
    return template

def load_coor(coordinates, template):
    """
        Create a template with new coordinates
        Input: output xyz coordinates and template without coordinates 
        Output: template with new coordinates
    """
    cur_cor = None
    cur_cor = template[:]
    index = 0
    for i in range(len(coordinates) // 3):
        x, y, z = coordinates[3 * i: 3 * (i + 1)]
        x, y, z = "%.3f" % x, "%.3f" % y, "%.3f" % z
        for cor, end in zip([x, y, z], [38, 46, 54]):
            length = len(cor)
            cur_cor[index] = cur_cor[index][:end - length] \
                + cor + cur_cor[index][end:]

        index += 1
    return cur_cor

def write_file(pdb_file,cur_cor):
    """
        Write the coordinates into PDB
        Inputs: output file name and decoded template
    """
    file = open(pdb_file, "w")
    for line in cur_cor:
        file.write(line)
    file.close()
    return None

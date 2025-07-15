import importlib.resources as pkg_resources
from EnsembleAnalysis import database
from psfgen import PsfGen

# force field file
ff_path = pkg_resources.files(database).joinpath("top_all22_prot.rtf").__fspath__()

# now only consider one chain
def generate_psf_from_pdb(pdb, outpsf, N_ter_patch='NTER', C_ter_patch='CTER'):
    global ff_path
    gen = PsfGen()
    gen.read_topology(ff_path)

    # alias for HIS
    gen.alias_residue('HSE', 'HIS')

    segid='A'
    gen.add_segment(segid=segid, pdbfile=pdb)

    # specifial patches for PRO and GLY
    if gen.get_resname('A', 0) == 'PRO':
        gen.patch('PROP', [(segid, gen.get_resids(segid)[0]),])
    elif gen.get_resname('A', 0) == 'GLY':
        gen.patch('GLYP', [(segid, gen.get_resids(segid)[0]),])
    else:
        gen.patch(N_ter_patch, [(segid, gen.get_resids(segid)[0]),])
    gen.patch(C_ter_patch, [(segid, gen.get_resids(segid)[-1]),])

    gen.write_psf(outpsf)


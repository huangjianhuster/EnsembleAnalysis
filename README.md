# EnsembleAnalysis

A package for analyzing conformational ensemble of protein systems.

Features:

1. automate general analysis on protein conformational ensembles.
2. plot functions for visualization

# Dependencies and installation

The python version >= 3.10 (tested under 3.10, but newer versions should also work)

```bash
conda install numpy matplotlib scipy pandas mdtraj biopython mdanalysis
conda install conda-forge::psfgen
```


**Installation**

Download this repository and `cd` to the directory storing the `pyproject.toml` file.

For release:

```python
pip install .
```

For developer:

```python
pip install -e .
```

# Ensemble analysis examples

There are three classes defined in the `EnsembleAnalysis.core.ensemble` module:

1. `Ensemble`: for general ensemble analysis
2. `IdpEnsemble`: for intrinsically disorder protein ensembles
3. `FoldedEnsemble`: for folded protein ensembles

```python
from EnsembleAnalysis.core.ensemble import *

psf = "path/to/psf"
trj = "path/to/trj" # xtc, dcd etc format (should be MDAnalysis-friendly)
# optional
top = "path/to/top"

en = Ensemble(psf, trj)
# Basic information
sequence = en.sequence
resid = en.resid

# Bacic calculations
secondary_structure = en.get_ss()
end2end = en.get_end2end()
psi = en.get_psi()
psi_5_10 = en.get_psi('5-10')
phi = en.get_phi()
phi_5_10 = en.get_phi(''5-10)
rmsd = en.get_rmsd() # by default align to the first frame
rmsf = en.get_rmsf() # CA RMSF values (no alignment performed)
```

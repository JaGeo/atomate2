# TODO:
# We will generate a phonon flow and run it
# We will first test it with smaller kpoints and also set specific values for NPAR in each of the jobs

from atomate2.vasp.flows.phonons import PhononMaker
from jobflow import run_locally
from pymatgen.core import Structure

# construct a rock salt MgO structure
mgo_structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# make a band structure flow to optimise the structure and obtain the band structure
phonon_flow =PhononMaker().make(mgo_structure)

# run the job
run_locally(phonon_flow, create_folders=True)

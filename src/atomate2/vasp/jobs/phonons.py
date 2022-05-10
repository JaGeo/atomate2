from __future__ import annotations

import copy
import sys
import tempfile
import xml.parsers.expat
from pathlib import Path
from typing import List
import logging

from atomate2 import SETTINGS
from atomate2.vasp.jobs.base import BaseVaspMaker
# from fireworks import LaunchPad
from jobflow import Flow, Response, job
from monty.io import zopen
from phonopy import Phonopy
from phonopy.interface.vasp import Vasprun as phVasprun
from phonopy.interface.vasp import VasprunxmlExpat as phVasprunxmlExpat
from phonopy.interface.vasp import check_forces as phcheck_forces
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import elaborate_borns_and_epsilon
from phonopy.units import VaspToTHz
from pymatgen.analysis.elasticity import (
    Deformation,
)
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_ph_bs_symm_line
from pymatgen.io.phonopy import get_ph_dos
from pymatgen.io.phonopy import get_pmg_structure, get_phonopy_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation


logger = logging.getLogger(__name__)


__all__ = [
    "generate_phonon_displacements",
    "run_phonon_displacements",
    "generate_frequencies_eigenvectors",
    "PhononDisplacemetMaker"
]


# should be replaced by the phonopy functions later on. They can also work with gzipped files as well
def get_born_vasprunxml(
        filename="vasprun.xml",
        primitive_matrix=None,
        supercell_matrix=None,
        is_symmetry=True,
        symmetrize_tensors=False,
        symprec=1e-5,
):
    """Parse vasprun.xml to get NAC parameters.

    In phonopy, primitive cell is created through the path of
    unit cell -> supercell -> primitive cell. To trace this path exactly,
    `primitive_matrix` and `supercell_matrix` can be given, but these are
    optional.

    Returns
    -------


    """
    with zopen(filename, "rb") as f:
        vasprun = phVasprunxmlExpat(f)
        try:
            vasprun.parse()
        except xml.parsers.expat.ExpatError:
            raise xml.parsers.expat.ExpatError(
                'Could not parse "%s". Please check the content.' % filename
            )
        except ValueError:
            raise ValueError(
                'Could not parse "%s". Please check the content.' % filename
            )

    return elaborate_borns_and_epsilon(
        vasprun.cell,
        vasprun.born,
        vasprun.epsilon,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        is_symmetry=is_symmetry,
        symmetrize_tensors=symmetrize_tensors,
        symprec=symprec,
    )


def parse_set_of_forces(num_atoms, forces_filenames, use_expat=True, verbose=True):
    """Parse sets of forces of files."""
    if verbose:
        sys.stdout.write("counter (file index): ")

    count = 0
    is_parsed = True
    force_sets = []
    force_files = forces_filenames

    for filename in force_files:
        with zopen(filename, "rb") as fp:
            if verbose:
                sys.stdout.write("%d " % (count + 1))
            vasprun = phVasprun(fp, use_expat=use_expat)
            try:
                forces = vasprun.read_forces()
            except (RuntimeError, ValueError, xml.parsers.expat.ExpatError) as err:
                msg = (
                        'Could not parse "%s". Probably this vasprun.xml '
                        "is broken or some value diverges. Check this "
                        "calculation carefully before sending questions to the "
                        "phonopy mailing list." % filename
                )
                raise RuntimeError(msg) from err
            force_sets.append(forces)
            count += 1

            if not phcheck_forces(force_sets[-1], num_atoms, filename):
                is_parsed = False

    if verbose:
        print("")

    if is_parsed:
        return force_sets
    else:
        return []


def get_phonon_object(conventional, displacement, min_length, structure, sym_reduce, symprec):
    if conventional:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        structure = sga.get_conventional_standard_structure()
    transformation = CubicSupercellTransformation(min_length=min_length)
    transformation.apply_transformation(structure=structure)
    supercell_matrix = transformation.transformation_matrix.tolist()
    cell = get_phonopy_structure(structure)
    phonon = Phonopy(cell,
                     supercell_matrix,
                     primitive_matrix=[[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]],
                     factor=VaspToTHz,
                     symprec=symprec,
                     is_symmetry=sym_reduce)
    phonon.generate_displacements(distance=displacement)
    return phonon


# check if this can also be replaced with something better
def get_kpath(structure: Structure,  **kpath_kwargs):
    """
    get high-symmetry points in k-space
    Args:
        structure: Structure Object
    Returns:
    """
    highsymmkpath = HighSymmKpath(structure, **kpath_kwargs)
    kpath = highsymmkpath.kpath
    path = copy.deepcopy(kpath["path"])

    for ilabelset, labelset in enumerate(kpath["path"]):
        for ilabel, label in enumerate(labelset):
            path[ilabelset][ilabel] = kpath["kpoints"][label]
    return kpath["kpoints"], path

# TODO: check all parameters again
@job
def generate_phonon_displacements(
        structure: Structure,
        displacement: float = 0.01,
        min_length: float = 4.0,
        conventional: bool = False,
        symprec: float = SETTINGS.SYMPREC,
        sym_reduce: bool = True,
):
    """
    Generate elastic deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure object.
    displacement : float
        The displacement to be applied to the structure.
    min_length  : float
        minimum supercell size.
    conventional : bool
        Whether to use the conventional cell.
    symprec : float
        The symprec to use for the spacegroup analyzer.
    sym_reduce : bool
        Whether to reduce the symmetry of the structure.

    Returns
    -------
    List[Deformation]
        A list of diplacements.
    """
    phonon = get_phonon_object(conventional, displacement, min_length, structure, sym_reduce, symprec)
    supercells = phonon.supercells_with_displacements

    displacements = []
    for cell in supercells:
        displacements.append(get_pmg_structure(cell))
    return displacements


@job(output_schema=PhononBSDOSDoc)
def generate_frequencies_eigenvectors(
        structure: Structure,
        displacement_data: list[dict],
        born_data: str | Path = None,
        displacement: float = 0.01,
        min_length: float = 4.0,
        conventional: bool = False,
        symprec: float = SETTINGS.SYMPREC,
        sym_reduce: bool = True,
        npoints_band: int = 100,
        kpoint_density_dos: int = 7000,
):
    """
    Compute phonon band structures and density of states.

    Parameters
    ----------

    """
    # get phonon object from phonopy with correct settings again

    phonon = get_phonon_object(conventional, displacement, min_length, structure, sym_reduce, symprec)

    # do this file based even though force based would be better
    forces_filenames = []
    # Vasprunxml is missing each time
    for displacement in displacement_data:
        # decompress_file(str(Path(displacement["job_dir"]) / "vasprun.xml.gz").split(":")[1])
        forces_filenames.append(str(Path(displacement["job_dir"]) / "vasprun.xml.gz").split(":")[1])

    set_of_forces = parse_set_of_forces(num_atoms=get_pmg_structure(phonon.supercell).num_sites,
                                        forces_filenames=forces_filenames)

    # produce force constants
    # decompress_file(str((Path(born_data) / "vasprun.xml.gz").split(":")[1]))

    phonon.produce_force_constants(forces=set_of_forces)
    borns, epsilon, atom_indices = get_born_vasprunxml(str(Path(born_data) / "vasprun.xml.gz").split(":")[1],
                                                       primitive_matrix=phonon.primitive_matrix,
                                                       supercell_matrix=phonon.supercell_matrix)
    # compress_file(str(zpath(Path(born_data) / "vasprun.xml")))
    # get born charges from vasprun.xml

    phonon.nac_params = {"born": borns, "dielectric": epsilon, "factor": 14.400}

    # get phonon band structure
    tempfilename = tempfile.gettempprefix() + '.yaml'
    kpath_dict, kpath_concrete = get_kpath(structure)
    qpoints, connections = get_band_qpoints_and_path_connections(kpath_concrete, npoints=npoints_band)

    phonon.run_band_structure(qpoints, path_connections=connections)
    phonon.write_yaml_band_structure(
        filename=tempfilename)
    bs_symm_line = get_ph_bs_symm_line(tempfilename, labels_dict=kpath_dict)

    # get phonon density of states
    tempfilename = tempfile.gettempprefix() + '.yaml'
    kpoint = Kpoints.automatic_density(structure=structure, kppa=kpoint_density_dos, force_gamma=True)
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=tempfilename)
    dos = get_ph_dos(tempfilename)

    # get thermal properties
    # TODO: add computation of thermal properties as well

    # maybe, we can just give the folder and phonopy can create it?
    # do something to generate a phonon document
    phonon_doc = PhononBSDOSDoc(structure=structure, ph_bs=bs_symm_line, ph_dos=dos)
    print(phonon_doc)
    return phonon_doc


@job
def run_phonon_displacements(
        displacements,
        phonon_maker: BaseVaspMaker = None,
):
    """
    Run elastic deformations.

    Note, this job will replace itself with N relaxation calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure.
    deformations : list of Deformation
        The deformations to apply.
    prev_vasp_dir : str or Path or None
        A previous VASP directory to use for copying VASP outputs.
    phonon_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    """
    if phonon_maker is None:
        phonon_maker = PhononMaker()
    phonon_runs = []
    outputs = []
    for i, displacement in enumerate(displacements):
        phonon_job = phonon_maker.make(
            displacement
        )
        phonon_job.append_name(f" {i + 1}/{len(displacements)}")
        phonon_runs.append(phonon_job)

        # extract the outputs we want
        # maybe add forces as well later on
        output = {
            "displacement_number": i,
            "uuid": phonon_job.output.uuid,
            "job_dir": phonon_job.output.dir_name,
        }

        outputs.append(output)

    relax_flow = Flow(phonon_runs, outputs)
    return Response(replace=relax_flow)


@dataclass
class PhononDisplacemetMaker(BaseVaspMaker):
    """
    Maker to perform an static calculation as a part of the finite displacement method.

    The input set is for a static run with tighter convergence parameters. Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputSetGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "phonon static"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 100},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 3,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "EDIFFG": -0.001,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 0,
                "LCHARG": False,
                "ISMEAR": 0,
                "NPAR": 4,
            },
        )
    )

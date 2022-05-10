"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OnMissing
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker, generate_phonon_displacements, \
    run_phonon_displacements, generate_frequencies_and_eigenvectors


__all__ = ["PhononMaker"]


@dataclass
class PhononMaker(Maker):
    """
    Maker to calculate harmonic phonons with VASP and Phonopy.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization effects,
    a correction of the dynamical matrix based on BORN charges can be performed.
    Finally, phonon densities of states, phonon band structures and thermodynamic properties are computed.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too many displacement calculations
        will be generated.
        It is recommended to check the convergence parameters here and adjust them if necessary. The default might
        not be strict enough for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    #TODO: adapt these parameters


    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the reduction of symmetry.
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    phonon_displacement_maker : .BaseVaspMaker
        Maker used to compute the forces for a supercell.
    generate_phonon_displacements_kwargs : dict
        Keyword arguments passed to :obj:`generate_elastic_deformations`.
    compute_harmonic_phonons_kwargs : dict
        Keyword arguments passed to :obj:`fit_elastic_tensor`.
    """

    name: str = "phonon"
    #order: int = 2
    sym_reduce: bool = True
    symprec: float = SETTINGS.SYMPREC
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    #elastic_relax_maker: BaseVaspMaker = field(default_factory=ElasticRelaxMaker)
    #generate_elastic_deformations_kwargs: dict = field(default_factory=dict)
    #fit_elastic_tensor_kwargs: dict = field(default_factory=dict)

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate the elastic constant.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_vasp_dir = bulk.output.dir_name

        # generate the dipslacements
        # add all kwargs and other arguments this displacement maker
        displacements = generate_phonon_displacements(structure=structure)
        #
        # jobs.append(displacements)
        # # perform the phonon displacement calculations
        # vasp_displacement_calcs = run_phonon_displacements(displacements.output, phonon_maker=PhononMaker())
        #
        # jobs.append(vasp_displacement_calcs)
        #
        # # create a born set
        # born_set = StaticSetGenerator(user_incar_settings={"LEPSILON": True}, user_kpoints_settings={"grid_density": 100})
        # born_job = StaticMaker(input_set_generator=born_set).make(structure=relaxed_structure)
        #
        # jobs.append(born_job)
        #
        # # Collect all information from the previous jobs
        # # At the moment,
        # # I only transfer file paths so that phonopy can easily find the files, might be problematic if the files are stored on a different machine
        # phonon_collect = generate_frequencies_eigenvectors(structure=relaxed_structure,
        #                                                    displacement_data=vasp_displacement_calcs.output,
        #                                                    born_data=born_job.output.dir_name)
        #
        # jobs.append(phonon_collect)
        # # create a flow including all jobs for a phonon computation
        # my_flow = Flow(jobs, phonon_collect.output)

        #
        # deformations = generate_elastic_deformations(
        #     structure,
        #     order=self.order,
        #     sym_reduce=self.sym_reduce,
        #     symprec=self.symprec,
        #     **self.generate_elastic_deformations_kwargs,
        # )
        # vasp_deformation_calcs = run_elastic_deformations(
        #     structure,
        #     deformations.output,
        #     prev_vasp_dir=prev_vasp_dir,
        #     elastic_relax_maker=self.elastic_relax_maker,
        # )
        # fit_tensor = fit_elastic_tensor(
        #     structure,
        #     vasp_deformation_calcs.output,
        #     equilibrium_stress=equilibrium_stress,
        #     order=self.order,
        #     symprec=self.symprec if self.sym_reduce else None,
        #     **self.fit_elastic_tensor_kwargs,
        # )
        #
        # # allow some of the deformations to fail
        # fit_tensor.config.on_missing_references = OnMissing.NONE

        jobs += [deformations, vasp_deformation_calcs, fit_tensor]

        flow = Flow(
            jobs=jobs,
            output=fit_tensor.output,
            name=self.name,
        )
        return flow



"""Flows for Lobster computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.common.jobs import structure_to_conventional, structure_to_primitive
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, RelaxMaker
from atomate2.vasp.jobs.lobster import VaspLobsterMaker, get_vasp_lobster_jobs


__all__ = ["LobsterMaker"]



@dataclass
class LobsterMaker(Maker):
    """
    Maker to perform a Lobster computation

    Optional optimization.
    Optional static computation with symmetry
    static computation with ISYM=0 is performed
    several Lobster computations testing several basis sets is performed

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
        allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    code: str
        determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    """

    name: str = "lobster"
    # implemet different calculation types
    calculation_type: str = "standard",
    delete_all_wavecars: bool = True,
    user_lobsterin_settings: dict = None,
    user_incar_settings: dict = None,
    user_kpoints_settings: dict = None,
    user_supplied_basis: dict = None,
    isym: int = 0,
    additional_outputs: list[str] = None,
    additional_optimization: bool = False,
    additional_static_run: bool =True # will add an additional static run
    user_incar_settings_optimization: dict = None,
    user_kpoints_settings_optimization: dict = None,
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(RelaxMaker())
    )
    additional_statik_run_maker: BaseVaspMaker|None =field(
        default_factory=lambda : StaticMaker()
    )

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate bonding properties.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.

        """

        jobs = []

        # do a relaxation step first
        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            optimization_run_job_dir = bulk.output.dir_name
            optimization_run_uuid = bulk.output.uuid
        else:
            optimization_run_job_dir = None
            optimization_run_uuid = None



        # do a static WAVECAR computation with symmetry and standard number of bands first
        # Do a static VASP computation


        # find out maximum number of bands that needs to be computed
        # switch off symmetry?!
        #


        # do a WAVECAR computation with correct number of bands and create vasp jobs
        vaspjob=get_vasp_lobster_jobs(structure, address_max_basis=None, address_min_basis=None, prev_vasp_dir=None)
        jobs.append(vaspjob)

        # do Lobster computation with the previous WAVECAR
        # lobster job has to be implemented


        # delete wavecars


        flow = Flow(jobs, vaspjob.output)
        return flow

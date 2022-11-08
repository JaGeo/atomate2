"""Module defining amset jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, Response, job
from monty.serialization import loadfn
from monty.shutil import gzip_dir
from pymatgen.io.lobster import Lobsterin
from atomate2.lobster.files import copy_lobster_files, write_lobster_settings
from atomate2.lobster.schemas import LobsterTaskDocument
from atomate2.lobster.run import run_lobster

__all__ = ["PureLobsterMaker"]

logger = logging.getLogger(__name__)


@dataclass
class PureLobsterMaker(Maker):
    """
    LOBSTER job maker.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    resubmit : bool
        Maybe useful.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    """

    name: str = "lobster"
    resubmit: bool = False
    task_document_kwargs: dict = field(default_factory=dict)

    @job(output_schema=LobsterTaskDocument)
    def make(
        self,
        wavefunction_dir: str | Path = None,
        basis_dict: dict=None,
        # something for the basis
    ):
        """
        Run an LOBSTER calculation.

        Parameters
        ----------
        wavefunction_dir : str or Path
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster

        """
        # copy previous inputs # VASP for example
        copy_lobster_files(wavefunction_dir)


        # write lobster settings
        lobsterin=Lobsterin.standard_calculations_from_vasp_files("POSCAR","INCAR",dict_for_basis=basis_dict)
        lobsterin.write_lobsterin("lobsterin")
        # run lobster
        logger.info("Running LOBSTER")
        run_lobster()

        converged = None

        # parse amset outputs
        task_doc = LobsterTaskDocument.from_directory(
            Path.cwd(), **self.task_document_kwargs
        )
        task_doc.converged = converged

        # gzip folder
        gzip_dir(".")


        return Response(output=task_doc)

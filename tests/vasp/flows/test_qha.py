from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.qha import QhaMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
    update_user_kpoints_settings,
)


def test_qha(mock_vasp, clean_dir, si_diamond: Structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "EOS equilibrium relaxation": "Si_qha/EOS_equilibrum_relaxation",
        "static 1/1": "Si_qha/static_1_1",
        "static eos deformation 1": "Si_qha/static_eos_deformation_1",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "EOS equilibrium relaxation": {"incar_settings": ["NSW", "ISMEAR"]},
        "static 1/1": {"incar_settings": ["NSW", "ISMEAR"]},
        "static eos deformation 1": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    qha_maker = QhaMaker(
        number_of_frames=0,
        phonon_maker_kwargs={"min_length": 8, "born_maker": None},
        ignore_imaginary_modes=True,
        skip_analysis=True,
    ).make(structure=si_diamond)

    qha_maker = update_user_incar_settings(qha_maker, {"NPAR": 4, "ISMEAR": 0})
    qha_maker = update_user_kpoints_settings(
        qha_maker, kpoints_updates={"reciprocal_density": 10}
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(qha_maker, create_folders=True, ensure_success=True)
    assert len(responses) == 9
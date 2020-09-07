import logging

from simple_sagemaker.sm_project import SageMakerProject


def test_test(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_test")
    smProject = SageMakerProject(projectName="test")
    smProject = smProject

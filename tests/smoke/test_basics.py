import logging

import boto3


def test_project(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_project")

    from simple_sagemaker.sm_project import SageMakerProject

    sm_project = SageMakerProject(project_name="test")
    sm_project = sm_project


def test_task(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_task")

    from simple_sagemaker.sm_task import SageMakerTask

    boto3_session = boto3.Session()
    image_uri = None
    smTask = SageMakerTask(boto3_session, "taskName", image_uri, prefix="tests/smoke")
    smTask = smTask

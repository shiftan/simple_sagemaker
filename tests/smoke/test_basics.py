import logging

import boto3


def test_project(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_project")

    from simple_sagemaker.sm_project import SageMakerProject

    smProject = SageMakerProject(projectName="test")
    smProject = smProject


def test_task(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_task")

    from simple_sagemaker.sm_task import SageMakerTask

    boto3Session = boto3.Session()
    imageUri = None
    smTask = SageMakerTask(boto3Session, "taskName", imageUri, prefix="tests/smoke")
    smTask = smTask

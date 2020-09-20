import collections
import logging

import boto3
import sagemaker

from . import constants, iam_utils
from .ecr_sync import ECRSync
from .sm_task import SageMakerTask

logger = logging.getLogger(__name__)


class SageMakerProject:
    ImageParams = collections.namedtuple(
        "ImageParams",
        [
            "aws_repo_name",
            "repo_name",
            "img_tag",
            "docker_file_path_or_content",
            "framework",
            "version",
            "py_version",
        ],
    )
    CodeParams = collections.namedtuple(
        "CodeParams", ["source_dir", "entryPoint", "dependencies"]
    )
    InstanceParams = collections.namedtuple(
        "InstanceParams",
        [
            "instance_type",
            "instance_count",
            "volume_size",
            "use_spot_instances",
            "max_run",
            "maxWait",
        ],
    )
    # IOParams = collections.namedtuple("IOParams", ["input_data_path", "distribution", "model_uri"])

    def __init__(
        self,
        project_name,
        boto3_session=None,
        role_name=constants.DEFAULT_IAM_ROLE,
        bucket_name=None,
        smSession=None,
    ):
        self.project_name = project_name
        self.role_name = role_name
        self.tasks = {}

        if boto3_session is None:
            boto3_session = boto3.Session()
        self.boto3_session = boto3_session

        if smSession is None:
            smSession = sagemaker.Session(boto_session=boto3_session)
        self.smSession = smSession

        if not bucket_name:
            bucket_name = self.smSession.default_bucket()
        self.bucket_name = bucket_name
        # self.createBucket()

    def setDefaultImageParams(
        self,
        aws_repo_name=None,
        repo_name=None,
        img_tag=constants.DEFAULT_REPO_TAG,
        docker_file_path_or_content=None,
        framework="pytorch",
        version=None,
        py_version=None,
    ):
        self.defaultImageParams = SageMakerProject.ImageParams(
            aws_repo_name,
            repo_name,
            img_tag,
            docker_file_path_or_content,
            framework,
            version,
            py_version,
        )

    def setDefaultCodeParams(
        self, source_dir=None, entryPoint=None, dependencies=list()
    ):
        self.defaultCodeParams = SageMakerProject.CodeParams(
            source_dir, entryPoint, dependencies
        )

    def setDefaultInstanceParams(
        self,
        instance_type=constants.DEFAULT_INSTANCE_TYPE,
        instance_count=constants.DEFAULT_INSTANCE_COUNT,
        volume_size=constants.DEFAULT_VOLUME_SIZE,
        use_spot_instances=constants.DEFAULT_USE_SPOT,
        max_run=constants.DEFAULT_MAX_RUN,
        maxWait=constants.DEFAULT_MAX_WAIT,
    ):
        self.defaultInstanceParams = SageMakerProject.InstanceParams(
            instance_type,
            instance_count,
            volume_size,
            use_spot_instances,
            max_run,
            maxWait,
        )

    def createBucket(self):
        client = self.boto3_session.client("s3")

        if "us-east-1" != self.boto3_session.region_name:
            location = {"LocationConstraint": self.boto3_session.region_name}
            client.create_bucket(
                Bucket=self.bucket_name, CreateBucketConfiguration=location
            )
        else:
            client.create_bucket(Bucket=self.bucket_name)

    def createIAMRole(self):
        iam_utils.createSageMakerIAMRole(self.boto3_session, self.role_name)

    def allowAccessToS3Bucket(
        self, bucket_name, policy_name=constants.DEFAULT_IAM_BUCKET_POLICY
    ):
        iam_utils.allowAccessToS3Bucket(
            self.boto3_session, self.role_name, policy_name, bucket_name
        )

    def addTask(self, task_name, smTask):
        assert task_name not in self.tasks, f"{task_name} already exists!"
        self.tasks[task_name] = smTask

    def buildOrGetImage(self, instance_type, **kwargs):
        args = self.defaultImageParams._replace(**kwargs)
        dockerSync = ECRSync(self.boto3_session)
        image_uri = dockerSync.buildAndPushDockerImage(
            instance_type=instance_type, **(args._asdict())
        )
        return image_uri

    def runTask(
        self,
        task_name,
        image_uri,
        hyperparameters,
        input_data_path=None,
        clean_state=False,
        forceRunning=False,
        tags=[],
        **kwargs,
    ):
        assert task_name not in self.tasks, f"{task_name} already exists!"
        smTask = SageMakerTask(
            self.boto3_session,
            task_name,
            image_uri,
            self.project_name,
            self.bucket_name,
            smSession=self.smSession,
        )
        if input_data_path:
            smTask.uploadOrSetInputData(input_data_path)
        args = self.defaultCodeParams._asdict()
        args.update(self.defaultInstanceParams._asdict())

        args.update(kwargs)

        tags.append({"Key": "SimpleSagemakerProject", "Value": self.project_name})
        import inspect

        tags.append(
            {
                "Key": "SimpleSagemakerCallingModule",
                "Value": inspect.stack()[1].filename,
            }
        )

        if clean_state:
            smTask.clean_state()

        job_name = None if forceRunning else self.getCompletionJobName(task_name)
        if job_name:
            logger.info(f"Task {task_name} is already completed by {job_name}")
            smTask.bindToJob(job_name)
        else:
            job_name = smTask.runTrainingJob(
                self.defaultImageParams.framework,
                role_name=self.role_name,
                hyperparameters=hyperparameters,
                tags=tags,
                **args,
            )

        self.addTask(task_name, smTask)
        return smTask, job_name

    def cleanFolder(self):
        s3c = self.boto3_session.client("s3")
        for file in self.smSession.list_s3_files(self.bucket_name, self.project_name):
            s3c.delete_object(Bucket=self.bucket_name, Key=file)

    def getInputConfig(
        self,
        task_name,
        distribution="FullyReplicated",
        model=False,
        output=False,
        state=False,
    ):
        if task_name in self.tasks:
            smTask = self.tasks[task_name]
        else:
            smTask = SageMakerTask(
                self.boto3_session,
                task_name,
                None,
                self.project_name,
                self.bucket_name,
                smSession=self.smSession,
            )
            job_name = self.getCompletionJobName(task_name)
            assert job_name, f"Task {task_name} isn't completed!"
            smTask.bindToJob(job_name)

        return smTask.getInputConfig(distribution, model, output, state)

    def downloadResults(
        self,
        task_name,
        outputBase,
        logs=True,
        state=True,
        model=True,
        output=True,
        source=True,
    ):
        smTask = self.tasks[task_name]
        return smTask.downloadResults(
            outputBase,
            logs=logs,
            state=state,
            model=model,
            output=output,
            source=source,
        )

    def _getS3Subdirs(self, bucket, prefix):
        client = self.boto3_session.client("s3")
        result = client.list_objects(Bucket=bucket, Prefix=prefix + "/", Delimiter="/")
        if "CommonPrefixes" not in result:
            return list()
        return [
            commonPreifx["Prefix"].split("/")[-2]
            for commonPreifx in result["CommonPrefixes"]
        ]

    def getCompletionStatus(self, task_name):
        taskS3Uri = SageMakerTask.getStateS3Uri(
            self.bucket_name, self.project_name, task_name
        )
        (bucket, key) = sagemaker.s3.parse_s3_url(taskS3Uri)
        subdirs = self._getS3Subdirs(bucket, key)
        results = dict.fromkeys(subdirs)
        for subdir in subdirs:
            try:
                completedContent = self.smSession.read_s3_file(
                    bucket, sagemaker.s3.s3_path_join(key, subdir, "__COMPLETED__")
                )
                results[subdir] = completedContent
            except:  # noqa: E722
                logger.warning(f"Couldn't get completion status for {subdir}")
        return results

    def getCompletionJobName(self, task_name):
        completionResults = self.getCompletionStatus(task_name)
        values = list(set(completionResults.values()))
        if len(values) == 1 and values[0] is not None:
            return values[0]
        return False

import collections
import json
import logging

import boto3
import sagemaker

from .ecr_sync import ECRSync
from .sm_task import SageMakerTask

logger = logging.getLogger(__name__)


class SageMakerProject:
    ImageParams = collections.namedtuple(
        "ImageParams", ["awsRepoName", "repoName", "imgTag", "dockerFilePathOrContent"]
    )
    CodeParams = collections.namedtuple(
        "CodeParams", ["sourceDir", "entryPoint", "dependencies"]
    )
    InstanceParams = collections.namedtuple(
        "InstanceParams",
        [
            "instanceType",
            "instanceCount",
            "volumeSize",
            "useSpotInstances",
            "maxRun",
            "maxWait",
        ],
    )
    # IOParams = collections.namedtuple("IOParams", ["inputDataPath", "distribution", "modelUri"])

    def __init__(
        self,
        projectName,
        boto3Session=None,
        roleName="SageMakerIAMRole",
        bucketName=None,
        smSession=None,
    ):
        self.projectName = projectName
        self.roleName = roleName
        self.tasks = {}

        if boto3Session is None:
            boto3Session = boto3.Session()
        self.boto3Session = boto3Session

        if smSession is None:
            smSession = sagemaker.Session(boto_session=boto3Session)
        self.smSession = smSession

        if not bucketName:
            bucketName = self.smSession.default_bucket()
        self.bucketName = bucketName
        # self.createBucket()

    def setDefaultImageParams(
        self, awsRepoName, repoName, imgTag, dockerFilePathOrContent
    ):
        self.defaultImageParams = SageMakerProject.ImageParams(
            awsRepoName, repoName, imgTag, dockerFilePathOrContent
        )

    def setDefaultCodeParams(self, sourceDir, entryPoint, dependencies):
        self.defaultCodeParams = SageMakerProject.CodeParams(
            sourceDir, entryPoint, dependencies
        )

    def setDefaultInstanceParams(
        self,
        instanceType="ml.m5.large",
        instanceCount=1,
        volumeSize=30,
        useSpotInstances=True,
        maxRun=24 * 60 * 60,
        maxWait=24 * 60 * 60,
    ):
        self.defaultInstanceParams = SageMakerProject.InstanceParams(
            instanceType, instanceCount, volumeSize, useSpotInstances, maxRun, maxWait
        )

    def createBucket(self):
        client = self.boto3Session.client("s3")

        if "us-east-1" != self.boto3Session.region_name:
            location = {"LocationConstraint": self.boto3Session.region_name}
            client.create_bucket(
                Bucket=self.bucketName, CreateBucketConfiguration=location
            )
        else:
            client.create_bucket(Bucket=self.bucketName)

    def createIAMRole(self):
        logger.info(
            f"Creating SageMaker IAM Role: {self.roleName} with an attached AmazonSageMakerFullAccess policy..."
        )

        trustRelationship = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "",
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        client = self.boto3Session.client("iam")
        try:
            client.get_role(RoleName=self.roleName)
        except:  # noqa: E722
            client.create_role(
                RoleName=self.roleName,
                AssumeRolePolicyDocument=json.dumps(trustRelationship),
            )
        response = client.attach_role_policy(
            RoleName=self.roleName,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )
        assert (
            response["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), f"Couldn't attach AmazonSageMakerFullAccess policy to role {self.roleName}"

    def addTask(self, taskName, smTask):
        assert taskName not in self.tasks, f"{taskName} already exists!"
        self.tasks[taskName] = smTask

    def buildOrGetImage(self, instanceType, **kwargs):
        args = self.defaultImageParams._replace(**kwargs)
        dockerSync = ECRSync(self.boto3Session)
        imageUri = dockerSync.buildAndPushDockerImage(
            instanceType=instanceType, **(args._asdict())
        )
        return imageUri

    def runTask(
        self,
        taskName,
        imageUri,
        hyperparameters,
        inputDataPath=None,
        cleanState=False,
        forceRunning=False,
        **kwargs,
    ):
        assert taskName not in self.tasks, f"{taskName} already exists!"
        smTask = SageMakerTask(
            self.boto3Session,
            taskName,
            imageUri,
            self.projectName,
            self.bucketName,
            smSession=self.smSession,
        )
        if inputDataPath:
            smTask.uploadOrSetInputData(inputDataPath)
        args = self.defaultCodeParams._asdict()
        args.update(self.defaultInstanceParams._asdict())

        args.update(kwargs)

        if cleanState:
            smTask.cleanState()

        jobName = None if forceRunning else self.getCompletionJobName(taskName)
        if jobName:
            logger.info(f"Task {taskName} is already completed by {jobName}")
            smTask.bindToJob(jobName)
        else:
            jobName = smTask.runTrainingJob(
                roleName=self.roleName, hyperparameters=hyperparameters, **args
            )

        self.addTask(taskName, smTask)
        return smTask, jobName

    def cleanFolder(self):
        s3c = self.boto3Session.client("s3")
        for file in self.smSession.list_s3_files(self.bucketName, self.projectName):
            s3c.delete_object(Bucket=self.bucketName, Key=file)

    def getInputConfig(
        self,
        taskName,
        distribution="FullyReplicated",
        model=False,
        output=False,
        state=False,
    ):
        smTask = self.tasks[taskName]
        return smTask.getInputConfig(distribution, model, output, state)

    def downloadResults(
        self,
        taskName,
        outputBase,
        logs=True,
        state=True,
        model=True,
        output=True,
        source=True,
    ):
        smTask = self.tasks[taskName]
        return smTask.downloadResults(
            outputBase,
            logs=logs,
            state=state,
            model=model,
            output=output,
            source=source,
        )

    def _getS3Subdirs(self, bucket, prefix):
        client = self.boto3Session.client("s3")
        result = client.list_objects(Bucket=bucket, Prefix=prefix + "/", Delimiter="/")
        if "CommonPrefixes" not in result:
            return list()
        return [
            commonPreifx["Prefix"].split("/")[-2]
            for commonPreifx in result["CommonPrefixes"]
        ]

    def getCompletionStatus(self, taskName):
        taskS3Uri = SageMakerTask.getStateS3Uri(
            self.bucketName, self.projectName, taskName
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

    def getCompletionJobName(self, taskName):
        completionResults = self.getCompletionStatus(taskName)
        values = list(set(completionResults.values()))
        if len(values) == 1 and values[0] is not None:
            return values[0]
        return False

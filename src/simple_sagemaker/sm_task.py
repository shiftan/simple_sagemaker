import logging
import os
import random
import string
import tarfile
from time import gmtime, strftime

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingOutput, ScriptProcessor
from sagemaker.pytorch.estimator import PyTorch

from . import constants
from .s3_sync import S3Sync

logger = logging.getLogger(__name__)


class SageMakerTask:
    """
    An easy to use wrapper around SageMaker jobs into Tasks, a collection of jobs sharing
        the same input data and state (between runs).
    """

    def __init__(
        self, boto3Session, taskName, imageUri, prefix, bucketName=None, smSession=None
    ):
        """
        Initializes a task

        Arguments:
            taskName -
            imageUri -
            bucketName - The default SageMaker's bucker is used if not provided
            smSession -

        Data is maintained on [bucketName]/[taskName]
        """
        self.boto3Session = boto3Session
        self.taskName = taskName
        self.imageUri = imageUri
        self.estimators = list()
        self.jobNames = list()

        if smSession is None:
            smSession = sagemaker.Session(boto_session=boto3Session)
        self.smSession = smSession

        if not bucketName:
            bucketName = self.smSession.default_bucket()
        self.bucketName = bucketName

        self.baseTaskS3Uri = SageMakerTask.getBaseTaskS3Uri(
            bucketName, prefix, taskName
        )
        self.stateS3Uri = SageMakerTask.getStateS3Uri(bucketName, prefix, taskName)
        self.stateLocalPath = constants.LOCAL_STATE_PATH
        self.inputS3Uri = None

        self.internalDependencies = [
            os.path.abspath(os.path.join(os.path.split(__file__)[0], "task_toolkit"))
        ]

    @staticmethod
    def getBaseTaskS3Uri(bucketName, prefix, taskName):
        return sagemaker.s3.s3_path_join("s3://", bucketName, prefix, taskName)

    @staticmethod
    def getStateS3Uri(bucketName, prefix, taskName):
        baseTaskS3Uri = SageMakerTask.getBaseTaskS3Uri(bucketName, prefix, taskName)
        return sagemaker.s3.s3_path_join(baseTaskS3Uri, "state")

    def _getJobName(self):
        timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        # Add a 8-bytes random string to avoid collisions
        randString = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        jobName = f"{self.taskName}-{timestamp_prefix}-{randString}"
        return jobName

    def __runProcessingJob(self, code, instanceType, instance_count, sagemaker_session):
        # TODO: fix refactoring
        assert False, "Should be fixed"
        script_processor = ScriptProcessor(
            command=["python3"],
            image_uri=self.imageUri,
            role=self.roleName,
            instance_count=instance_count,
            instance_type=instanceType,
            sagemaker_session=sagemaker_session,
        )

        jobName = self._getJobName()
        outputs = [ProcessingOutput(source="/opt/ml/processing/output")]
        script_processor.run(
            code=code,
            inputs=None,
            outputs=outputs,
            arguments=["aa", "bb"],
            job_name=jobName,
        )
        script_processor_job_description = script_processor.jobs[-1].describe()
        print(script_processor_job_description)
        return jobName

    def runTrainingJob(
        self,
        sourceDir,
        entryPoint,
        dependencies,
        hyperparameters,
        instanceType,
        instanceCount,
        roleName,
        additionalInputs=dict(),
        modelUri=None,
        useSpotInstances=False,
        maxWait=None,
        volumeSize=30,
        maxRun=24 * 60 * 60,
        distribution="FullyReplicated",
    ):
        """
        Runs a training job

        Arguments:
            sourceDir - local/s3
            entryPoint - entry point
            dependencies - additional local dependencies (directories) to be copied to the code path
            hyperparameters -
            instanceType -
            instanceCount -
            modelUri - local/s3
            ...

        Returns estimator object
        """
        logger.info(
            f"Running a training job sourceDir={sourceDir} entryPoint={entryPoint} hyperparameters={hyperparameters}..."
        )
        jobName = self._getJobName()

        # append the internal dependencies
        dependencies.extend(self.internalDependencies)

        pytorch_estimator = PyTorch(
            entry_point=entryPoint,
            source_dir=sourceDir,
            hyperparameters=hyperparameters,
            image_uri=self.imageUri,
            role=roleName,
            instance_count=instanceCount,
            instance_type=instanceType,
            sagemaker_session=self.smSession,
            checkpoint_s3_uri=self.stateS3Uri,
            checkpoint_local_path=self.stateLocalPath,
            output_path=self.baseTaskS3Uri,
            code_location=self.baseTaskS3Uri,
            # To check
            dependencies=dependencies,
            container_log_level=logging.INFO,
            volume_size=volumeSize,
            max_run=maxRun,
            model_uri=modelUri,
            use_spot_instances=useSpotInstances,
            max_wait=maxWait,
        )

        inputs = dict()
        if self.inputS3Uri:
            inputs.update(
                {"data": TrainingInput(self.inputS3Uri, distribution=distribution)}
            )
        if additionalInputs:
            inputs.update(additionalInputs)

        pytorch_estimator.fit(inputs=inputs if inputs else None, job_name=jobName)
        # training_job_description = pytorch_estimator.latest_training_job.describe()
        # logging.info(f"Job is done: {training_job_description}")

        self.estimators.append(pytorch_estimator)
        self.jobNames.append(jobName)
        return jobName

    def bindToJob(self, jobName):
        self.jobNames.append(jobName)

    def cleanState(self):
        uri = self.getOutputTargetUri(state=True)
        bucket, prefix = sagemaker.s3.parse_s3_url(uri)
        s3c = self.boto3Session.client("s3")
        for file in self.smSession.list_s3_files(bucket, prefix):
            s3c.delete_object(Bucket=bucket, Key=file)

    def uploadOrSetInputData(self, inputDataPath):
        """
        Use a local/s3 path as input data, uploads/sync to Task's input path if local path is given

        Arguments:
            inputDataPath - local/s3 path of input data
        """
        if inputDataPath.lower().startswith("s3://"):
            logger.info(f"Setting input data to {inputDataPath}...")
            self.inputS3Uri = inputDataPath
        else:
            # uploadedUri = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
            sync = S3Sync(self.boto3Session)
            self.inputS3Uri = sagemaker.s3.s3_path_join(self.baseTaskS3Uri, "input")
            logger.info(f"Syncing data from {inputDataPath} to {self.inputS3Uri}...")
            sync.syncFolderToS3(
                inputDataPath,
                self.bucketName,
                sagemaker.s3.parse_s3_url(self.inputS3Uri)[1],
            )

    def _downloadData(self, path, uri, extra_args):
        bucket, prefix = sagemaker.s3.parse_s3_url(uri)
        try:
            if self.smSession.list_s3_files(bucket, prefix):
                self.smSession.download_data(path, bucket, prefix, extra_args)
        except:  # noqa: E722
            logger.info(f"Couldn't download from {uri}", exc_info=True)

    def getOutputUri(self, jobName=None):
        """
        Get the output URI for a given / the last job
        """
        if jobName is None:
            jobName = self.jobNames[-1]
        return sagemaker.s3.s3_path_join(self.baseTaskS3Uri, jobName)

    def _extractTars(self, path):
        if not os.path.isdir(path):
            return
        for fileName in os.listdir(path):
            if fileName.endswith(".tar.gz"):
                tarFileName = os.path.join(path, fileName)
                if os.path.isfile(tarFileName):
                    tar = tarfile.open(tarFileName)
                    tar.extractall(path)
                    os.remove(tarFileName)

    def getOutputTargetUri(self, model=False, output=False, state=False, source=False):
        uri = None
        if model:
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), "output", "model.tar.gz"
            )
        elif output:
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), "output", "output.tar.gz"
            )
        elif state:
            uri = self.stateS3Uri
        elif source:
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), "source", "sourcedir.tar.gz"
            )
        return uri

    def getInputConfig(
        self, distribution="FullyReplicated", model=False, output=False, state=False
    ):
        uri = self.getOutputTargetUri(model=model, output=output, state=state)
        return TrainingInput(uri, distribution=distribution)

    def downloadResults(
        self,
        outputBase,
        logs=True,
        state=True,
        model=True,
        output=True,
        source=True,
        extractTars=True,
        extra_args=None,
    ):
        logger.info(f"Downloading results to {outputBase}")
        os.makedirs(outputBase, exist_ok=True)

        if logs:
            # get and save the logs
            logs = self.getLogs()
            logsPath = os.path.join(outputBase, "logs")
            os.makedirs(logsPath, exist_ok=True)
            for channelName in logs.keys():
                ff = open(os.path.join(logsPath, f"logs{channelName}"), "wt")
                for line in logs[channelName]:
                    ff.write(line)
                    ff.write("\n")
                ff.close()

        # download and extract state, output, model, source

        for (shouldDownload, argName) in zip(
            [state, model, output, source], ["state", "model", "output", "source"]
        ):
            if shouldDownload:
                outputPath = os.path.join(outputBase, argName)
                uri = self.getOutputTargetUri(**{argName: True})
                self._downloadData(outputPath, uri, extra_args)
                if extractTars:
                    if uri.endswith(".tar.gz"):
                        self._extractTars(outputPath)

    # Based on logs_for_job in https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/session.py
    def getLogs(self, jobName=None):
        """
        Get the logs for a given / the last job
        """
        if jobName is None:
            jobName = self.jobNames[-1]
        wait = False

        class logsWrapper(object):
            def __init__(self):
                self.logsChannels = dict()

            def __call__(self, index, s):
                self.logsChannels.setdefault(index, list()).append(s)

            def getChannels(self):
                return self.logsChannels.keys()

            def getLogsForChannel(self, channel):
                return self.logsChannels[channel]

        description = self.smSession.describe_training_job(jobName)
        (
            instance_count,
            stream_names,
            positions,
            client,
            log_group,
            dot,
            color_wrap,
        ) = sagemaker.session._logs_init(self.smSession, description, job="Training")
        lw = logsWrapper()  # raplace with our own class
        state = sagemaker.session._get_initial_job_state(  # noqa: F841
            description, "TrainingJobStatus", wait
        )
        sagemaker.session._flush_log_streams(
            stream_names, instance_count, client, log_group, jobName, positions, dot, lw
        )
        return lw.logsChannels


def main():
    print("...")


if __name__ == "__main__":
    main()


# Relevant documnets:
# https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html
# Docker SDK - https://docker-py.readthedocs.io/en/stable/images.html
# Amazon SageMaker ML Instance Types - https://aws.amazon.com/sagemaker/pricing/instance-types/
# Pricing, availability per processing type and region - https://aws.amazon.com/sagemaker/pricing/
# Instance Advisor - https://aws.amazon.com/ec2/spot/instance-advisor/

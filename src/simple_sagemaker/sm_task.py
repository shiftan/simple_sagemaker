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
from sagemaker.tensorflow.estimator import TensorFlow

from . import VERSION, constants
from .s3_sync import S3Sync

logger = logging.getLogger(__name__)


class SageMakerTask:
    """
    An easy to use wrapper around SageMaker jobs into Tasks, a collection of jobs sharing
        the same input data and state (between runs).
    """

    def __init__(
        self,
        boto3_session,
        task_name,
        image_uri,
        prefix,
        bucket_name=None,
        smSession=None,
    ):
        """
        Initializes a task

        Arguments:
            task_name -
            image_uri -
            bucket_name - The default SageMaker's bucker is used if not provided
            smSession -

        Data is maintained on [bucket_name]/[task_name]
        """
        self.boto3_session = boto3_session
        self.task_name = task_name
        self.image_uri = image_uri
        self.estimators = list()
        self.jobNames = list()

        if smSession is None:
            smSession = sagemaker.Session(boto_session=boto3_session)
        self.smSession = smSession

        if not bucket_name:
            bucket_name = self.smSession.default_bucket()
        self.bucket_name = bucket_name

        self.baseTaskS3Uri = SageMakerTask.getBaseTaskS3Uri(
            bucket_name, prefix, task_name
        )
        self.stateS3Uri = SageMakerTask.getStateS3Uri(bucket_name, prefix, task_name)
        self.stateLocalPath = constants.LOCAL_STATE_PATH
        self.inputS3Uri = None

        self.internalDependencies = [
            os.path.abspath(os.path.join(os.path.split(__file__)[0], "task_toolkit"))
        ]

    @staticmethod
    def getBaseTaskS3Uri(bucket_name, prefix, task_name):
        return sagemaker.s3.s3_path_join("s3://", bucket_name, prefix, task_name)

    @staticmethod
    def getStateS3Uri(bucket_name, prefix, task_name):
        baseTaskS3Uri = SageMakerTask.getBaseTaskS3Uri(bucket_name, prefix, task_name)
        return sagemaker.s3.s3_path_join(baseTaskS3Uri, "state")

    def _getJobName(self):
        timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        # Add a 8-bytes random string to avoid collisions
        randString = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        job_name = f"{self.task_name}-{timestamp_prefix}-{randString}"
        return job_name

    def __runProcessingJob(
        self, code, instance_type, instance_count, sagemaker_session
    ):
        # TODO: fix refactoring
        assert False, "Should be fixed"
        script_processor = ScriptProcessor(
            command=["python3"],
            image_uri=self.image_uri,
            role=self.role_name,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
        )

        job_name = self._getJobName()
        outputs = [ProcessingOutput(source="/opt/ml/processing/output")]
        script_processor.run(
            code=code,
            inputs=None,
            outputs=outputs,
            arguments=["aa", "bb"],
            job_name=job_name,
        )
        script_processor_job_description = script_processor.jobs[-1].describe()
        print(script_processor_job_description)
        return job_name

    def runTrainingJob(
        self,
        framework,
        source_dir,
        entry_point,
        dependencies,
        hyperparameters,
        instance_type,
        instance_count,
        role_name,
        additional_inputs=dict(),
        model_uri=None,
        use_spot_instances=False,
        max_wait=None,
        volume_size=30,
        max_run=24 * 60 * 60,
        tags=dict(),
        distribution="FullyReplicated",
        metric_definitions=dict(),
        enable_sagemaker_metrics=False,
        **additionalEstimatorArgs,
    ):
        """
        Runs a training job

        Arguments:
            source_dir - local/s3
            entry_point - entry point
            dependencies - additional local dependencies (directories) to be copied to the code path
            hyperparameters -
            instance_type -
            instance_count -
            model_uri - local/s3
            ...

        Returns estimator object
        """
        logger.info(
            f"Running a training job source_dir={source_dir} entry_point={entry_point} hyperparameters={hyperparameters}..."
        )
        job_name = self._getJobName()

        # append the internal dependencies
        dependencies.extend(self.internalDependencies)

        tags["SimpleSagemakerTask"] = self.task_name
        tags["SimpleSagemakerVersion"] = VERSION
        tags = [{"Key": k, "Value": v} for k, v in tags.items()]

        metric_definitions = [
            {"Name": k, "Regex": v} for k, v in metric_definitions.items()
        ]

        if not use_spot_instances:
            max_wait = 0

        classes = {
            "pytorch": PyTorch,
            "tensorflow": TensorFlow,
        }
        estimator_class = classes[framework]

        estimator = estimator_class(
            entry_point=entry_point,
            source_dir=source_dir,
            hyperparameters=hyperparameters,
            image_uri=self.image_uri,
            role=role_name,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=self.smSession,
            checkpoint_s3_uri=self.stateS3Uri,
            checkpoint_local_path=self.stateLocalPath,
            output_path=self.baseTaskS3Uri,
            code_location=self.baseTaskS3Uri,
            dependencies=dependencies,
            container_log_level=logging.INFO,
            volume_size=volume_size,
            max_run=max_run,
            model_uri=model_uri,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            tags=tags,
            metric_definitions=metric_definitions,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            **additionalEstimatorArgs,
        )
        inputs = dict()
        if self.inputS3Uri:
            inputs.update(
                {"data": TrainingInput(self.inputS3Uri, distribution=distribution)}
            )
        if additional_inputs:
            inputs.update(additional_inputs)

        estimator.fit(inputs=inputs if inputs else None, job_name=job_name)
        # training_job_description = estimator.latest_training_job.describe()
        # logging.info(f"Job is done: {training_job_description}")

        self.estimators.append(estimator)
        self.jobNames.append(job_name)
        return job_name

    def bindToJob(self, job_name):
        self.jobNames.append(job_name)

    def clean_state(self):
        uri = self.getOutputTargetUri(state=True)
        bucket, prefix = sagemaker.s3.parse_s3_url(uri)
        s3c = self.boto3_session.client("s3")
        for file in self.smSession.list_s3_files(bucket, prefix):
            s3c.delete_object(Bucket=bucket, Key=file)

    def uploadOrSetInputData(self, input_data_path):
        """
        Use a local/s3 path as input data, uploads/sync to Task's input path if local path is given

        Arguments:
            input_data_path - local/s3 path of input data
        """
        if input_data_path.lower().startswith("s3://"):
            logger.info(f"Setting input data to {input_data_path}...")
            self.inputS3Uri = input_data_path
        else:
            # uploadedUri = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
            sync = S3Sync(self.boto3_session)
            self.inputS3Uri = sagemaker.s3.s3_path_join(self.baseTaskS3Uri, "input")
            logger.info(f"Syncing data from {input_data_path} to {self.inputS3Uri}...")
            sync.syncFolderToS3(
                input_data_path,
                self.bucket_name,
                sagemaker.s3.parse_s3_url(self.inputS3Uri)[1],
            )

    def _downloadData(self, path, uri, extra_args):
        bucket, prefix = sagemaker.s3.parse_s3_url(uri)
        try:
            if self.smSession.list_s3_files(bucket, prefix):
                self.smSession.download_data(path, bucket, prefix, extra_args)
        except:  # noqa: E722
            logger.info(f"Couldn't download from {uri}", exc_info=True)

    def getOutputUri(self, job_name=None):
        """
        Get the output URI for a given / the last job
        """
        if job_name is None:
            job_name = self.jobNames[-1]
        return sagemaker.s3.s3_path_join(self.baseTaskS3Uri, job_name)

    def _extractTars(self, path):
        if not os.path.isdir(path):
            return
        for file_name in os.listdir(path):
            if file_name.endswith(".tar.gz"):
                tarFileName = os.path.join(path, file_name)
                if os.path.isfile(tarFileName):
                    tar = tarfile.open(tarFileName)
                    tar.extractall(path)
                    os.remove(tarFileName)

    def getOutputTargetUri(self, model=False, output=False, state=False, source=False):
        assert (
            model + output + state + source == 1
        ), "Onlt one output type flag should be set"
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
                self.getOutputUri(), "source", "source_dir.tar.gz"
            )
        return uri

    def getInputConfig(self, output_type, distribution="FullyReplicated"):
        uri = self.getOutputTargetUri(**{output_type: True})
        return TrainingInput(uri, distribution=distribution)

    def downloadResults(
        self,
        output_base,
        logs=True,
        state=True,
        model=True,
        output=True,
        source=True,
        extractTars=True,
        extra_args=None,
    ):
        logger.info(f"Downloading results to {output_base}")
        os.makedirs(output_base, exist_ok=True)

        if logs:
            # get and save the logs
            logs = self.getLogs()
            logsPath = os.path.join(output_base, "logs")
            os.makedirs(logsPath, exist_ok=True)
            for channel_name in logs.keys():
                ff = open(os.path.join(logsPath, f"logs{channel_name}"), "wt")
                for line in logs[channel_name]:
                    ff.write(line)
                    ff.write("\n")
                ff.close()

        # download and extract state, output, model, source

        for (shouldDownload, argName) in zip(
            [state, model, output, source], ["state", "model", "output", "source"]
        ):
            if shouldDownload:
                output_path = os.path.join(output_base, argName)
                uri = self.getOutputTargetUri(**{argName: True})
                self._downloadData(output_path, uri, extra_args)
                if extractTars:
                    if uri.endswith(".tar.gz"):
                        self._extractTars(output_path)

    # Based on logs_for_job in https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/session.py
    def getLogs(self, job_name=None):
        """
        Get the logs for a given / the last job
        """
        if job_name is None:
            job_name = self.jobNames[-1]
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

        description = self.smSession.describe_training_job(job_name)
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
            stream_names,
            instance_count,
            client,
            log_group,
            job_name,
            positions,
            dot,
            lw,
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

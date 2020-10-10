import logging
import os
import random
import string
import tarfile
from time import gmtime, strftime

import sagemaker
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)
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
        local_mode=False,
        task_type=None,
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
        self.sm_client = boto3_session.client("sagemaker")
        self.task_name = task_name
        self.image_uri = image_uri
        self.estimators = list()
        self.jobNames = list()
        self.descriptions = list()
        self.local_mode = local_mode
        self.task_type = task_type
        self.prefix = prefix

        if smSession is None:
            smSession = sagemaker.Session(boto_session=boto3_session)
        self.smSession = smSession

        if not bucket_name:
            bucket_name = self.smSession.default_bucket()
        self.bucket_name = bucket_name

        self.baseTaskS3Uri = SageMakerTask.getBaseTaskS3Uri(
            bucket_name, prefix, task_name
        )
        self.stateS3Uri = None
        self.stateLocalPath = None
        if not local_mode:
            self.stateS3Uri = SageMakerTask.getStateS3Uri(
                bucket_name, prefix, task_name
            )
            self.stateLocalPath = constants.LOCAL_STATE_PATH
        self.inputS3Uri = None

        self.internalDependencies = [
            os.path.abspath(os.path.join(os.path.split(__file__)[0], "worker_toolkit"))
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

    def runProcessing(
        self,
        entrypoint=None,
        command=None,
        env=None,
        code=None,
        arguments=None,
        inputs=list(),
        outputs=list(),
        instance_type=constants.DEFAULT_INSTANCE_TYPE_TRAINING,
        instance_count=constants.DEFAULT_INSTANCE_COUNT,
        role_name=constants.DEFAULT_IAM_ROLE,
        volume_size=constants.DEFAULT_VOLUME_SIZE,
        max_run_mins=constants.DEFAULT_MAX_RUN,
        tags=dict(),
        input_distribution="FullyReplicated",
        dependencies=list(),
    ):
        logger.info(
            f"===== Running a processing job {self.task_name} entrypoint={entrypoint} "
            f"command={command} code={code} arguments={arguments}... ====="
        )
        job_name = self._getJobName()

        # ## Outputs

        # state - continuesly updated
        state_path = "/opt/ml/processing/state"
        outputs.append(
            ProcessingOutput(state_path, self.stateS3Uri, "state", "Continuous")
        )
        env["SSM_STATE"] = state_path

        # output - copied by end of job
        output_path = "/opt/ml/processing/output"
        output_s3_uri = sagemaker.s3.s3_path_join(
            self.baseTaskS3Uri, job_name, "output"
        )
        outputs.append(
            ProcessingOutput(output_path, output_s3_uri, "output", "EndOfJob")
        )
        env["SSM_OUTPUT"] = output_path

        # ## Inputs

        # prev state
        bucket, prefix = sagemaker.s3.parse_s3_url(self.stateS3Uri)
        if self.smSession.list_s3_files(bucket, prefix):
            prev_state_path = "/opt/ml/processing/state_prev"
            inputs.append(
                ProcessingInput(
                    self.stateS3Uri,
                    prev_state_path,
                    "state_prev",
                    s3_data_distribution_type="FullyReplicated",
                )
            )

        # dependencies

        # append the internal dependencies
        dependencies.extend(self.internalDependencies)
        for dep in dependencies:
            dep = os.path.abspath(dep)
            basename = os.path.basename(dep)
            local_path = f"/opt/ml/processing/input/code/{basename}"
            inputs.append(
                ProcessingInput(
                    dep,
                    local_path,
                    "DEP_" + basename,
                    s3_data_distribution_type="FullyReplicated",
                )
            )

        # input data
        if self.inputS3Uri:
            data_path = "/opt/ml/processing/data"
            inputs.append(
                ProcessingInput(
                    self.inputS3Uri,
                    data_path,
                    "data",
                    s3_data_distribution_type=input_distribution,
                )
            )
            env["SM_CHANNEL_DATA"] = data_path

        tags["SimpleSagemakerTask"] = self.task_name
        tags["SimpleSagemakerVersion"] = VERSION
        tags = [{"Key": k, "Value": v} for k, v in tags.items()]

        additional_args = dict()
        if code:
            processor_class = ScriptProcessor
            additional_args["command"] = command
        else:
            assert (
                not command
            ), "Command can't be given where code isn't given (for the `Processor` class)"
            processor_class = Processor
            additional_args["entrypoint"] = entrypoint

        processor = processor_class(
            role=role_name,
            image_uri=self.image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size,
            max_runtime_in_seconds=max_run_mins * 60,
            sagemaker_session=self.smSession,
            tags=tags,
            env=env,
            **additional_args,
        )
        if code:
            processor.run(
                code=code,
                inputs=inputs,
                outputs=outputs,
                arguments=arguments,
                job_name=job_name,
            )
        else:
            processor.run(
                inputs=inputs,
                outputs=outputs,
                arguments=arguments,
                job_name=job_name,
            )

        proecessing_job_description = self.smSession.describe_processing_job(job_name)

        self.estimators.append(processor)
        self.jobNames.append(job_name)
        self.descriptions.append(proecessing_job_description)
        # print(proecessing_job_description)
        # if "Completed" != proecessing_job_description["TrainingJobStatus"]:
        #    logger.error(
        #        f"Task failed with status: {proecessing_job_description['TrainingJobStatus']}"
        #    )
        return job_name

    def runTrainingJob(
        self,
        framework,
        source_dir,
        entry_point,
        dependencies,
        hyperparameters,
        instance_type=constants.DEFAULT_INSTANCE_TYPE_TRAINING,
        instance_count=constants.DEFAULT_INSTANCE_COUNT,
        role_name=constants.DEFAULT_IAM_ROLE,
        additional_inputs=dict(),
        model_uri=None,
        use_spot_instances=constants.DEFAULT_USE_SPOT,
        max_wait_mins=constants.DEFAULT_MAX_WAIT,
        volume_size=constants.DEFAULT_VOLUME_SIZE,
        max_run_mins=constants.DEFAULT_MAX_RUN,
        tags=dict(),
        input_distribution="FullyReplicated",
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
            f"===== Running a training job {self.task_name} source_dir={source_dir} "
            f"entry_point={entry_point} hyperparameters={hyperparameters}... ====="
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

        # zero max_wait_mins if not using spot instances
        if not use_spot_instances:
            max_wait_mins = 0
        # if using spot, and max_wait_mins isn't specified -> set it to max_run_mins
        elif not max_wait_mins:
            max_wait_mins = max_run_mins

        classes = {
            "pytorch": PyTorch,
            "tensorflow": TensorFlow,
        }
        estimator_class = classes[framework]

        # Configure TensorBoard
        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=self.baseTaskS3Uri,
            container_local_output_path="/opt/ml/output/tensorboard/",
        )

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
            max_run=max_run_mins * 60,
            model_uri=model_uri,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait_mins * 60,
            tags=tags,
            metric_definitions=metric_definitions,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            tensorboard_output_config=tensorboard_output_config,
            debugger_hook_config=False,
            **additionalEstimatorArgs,
        )
        inputs = dict()
        if self.inputS3Uri:
            inputs.update(
                {
                    "data": TrainingInput(
                        self.inputS3Uri, distribution=input_distribution
                    )
                }
            )
        if additional_inputs:
            inputs.update(additional_inputs)

        estimator.fit(inputs=inputs if inputs else None, job_name=job_name)
        # training_job_description = estimator.latest_training_job.describe()
        # logging.info(f"Job is done: {training_job_description}")
        training_job_description = self.smSession.describe_training_job(job_name)

        self.estimators.append(estimator)
        self.jobNames.append(job_name)
        self.descriptions.append(training_job_description)
        if "Completed" != training_job_description["TrainingJobStatus"]:
            logger.error(
                f"Task failed with status: {training_job_description['TrainingJobStatus']}"
            )
        return job_name

    def _getJobByName(self, name_contains):
        funcs_type = (
            (
                self.sm_client.list_training_jobs,
                constants.TASK_TYPE_TRAINING,
                "TrainingJobSummaries",
            ),
            (
                self.sm_client.list_processing_jobs,
                constants.TASK_TYPE_PROCESSING,
                "ProcessingJobSummaries",
            ),
        )
        for (list_func, task_type, results_field_name) in funcs_type:
            extra_args = {}
            while True:
                resp = list_func(
                    NameContains=name_contains, MaxResults=100, **extra_args
                )

                if resp.get(results_field_name, None):
                    assert (
                        len(resp[results_field_name]) == 1
                    ), f"More than one options for {name_contains}"
                    return resp[results_field_name][0], task_type

                if "NextToken" in resp:
                    extra_args["NextToken"] = resp["NextToken"]
                else:
                    break
        return None, None

    @staticmethod
    def getLastTrainingJob(sm_client, project_name, task_name):
        # look for training jobs
        search_res = sm_client.search(
            Resource="TrainingJob",
            SearchExpression={
                "Filters": [
                    {
                        "Name": "Tags.SimpleSagemakerTask",
                        "Operator": "Equals",
                        "Value": task_name,
                    },
                    {
                        "Name": "Tags.SimpleSagemakerProject",
                        "Operator": "Equals",
                        "Value": project_name,
                    },
                ]
            },
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        if search_res["Results"]:
            training_job = search_res["Results"][0]["TrainingJob"]
            status = training_job["TrainingJobStatus"]
            name = training_job["TrainingJobName"]
            time = training_job["CreationTime"]
            return name, status, time
        return None, None, None

    @staticmethod
    def getLastProcessingJob(boto3_session, sm_client, project_name, task_name):
        # look for processing jobs
        extra_args = {}
        arn_tags = {}
        rt_client = boto3_session.client("resourcegroupstaggingapi")
        while True:
            resp = rt_client.get_resources(
                ResourcesPerPage=100,
                TagFilters=[
                    {
                        "Key": "SimpleSagemakerProject",
                        "Values": [project_name],
                    },
                    {
                        "Key": "SimpleSagemakerTask",
                        "Values": [task_name],
                    },
                ],
                ResourceTypeFilters=["sagemaker:processing-job"],
                **extra_args,
            )

            for res_tags in resp["ResourceTagMappingList"]:
                tags = {x["Key"]: x["Value"] for x in res_tags["Tags"]}
                arn_tags[res_tags["ResourceARN"]] = tags

            if resp.get("PaginationToken", None):
                extra_args["PaginationToken"] = resp["PaginationToken"]
            else:
                break

        extra_args = {}
        while True:
            resp = sm_client.list_processing_jobs(
                NameContains=task_name,
                MaxResults=100,
                SortBy="CreationTime",
                SortOrder="Descending",
                **extra_args,
            )

            if resp.get("ProcessingJobSummaries", None):
                for job_summary in resp["ProcessingJobSummaries"]:
                    tags = arn_tags.get(job_summary["ProcessingJobArn"])
                    if (
                        tags
                        and tags.get("SimpleSagemakerProject", None) == project_name
                        and tags.get("SimpleSagemakerTask", None) == task_name
                    ):
                        return (
                            job_summary["ProcessingJobName"],
                            job_summary["ProcessingJobStatus"],
                            job_summary["CreationTime"],
                        )

            if "NextToken" in resp:
                extra_args["NextToken"] = resp["NextToken"]
            else:
                break
        return None, None, None

    @staticmethod
    def getLastJob(boto3_session, project_name, task_name, task_type=None):
        # Look for training job first and return it if it's there
        from botocore.config import Config

        config = Config(retries={"max_attempts": 10, "mode": "standard"})
        client = boto3_session.client("sagemaker", config=config)

        name, status, job_type = None, None, None

        if not task_type or task_type == constants.TASK_TYPE_TRAINING:
            name, status, time = SageMakerTask.getLastTrainingJob(
                client, project_name, task_name
            )
            if name:
                job_type = constants.TASK_TYPE_TRAINING

        if not task_type or task_type == constants.TASK_TYPE_PROCESSING:
            name2, status2, time2 = SageMakerTask.getLastProcessingJob(
                boto3_session, client, project_name, task_name
            )
            if name2:
                if not name or time2 > time:
                    name, status, time = name2, status2, time2
                    job_type = constants.TASK_TYPE_PROCESSING

        return name, job_type, status

    def bindToLastJob(self, job_name, task_type):
        self.task_type = task_type
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
        ), "Only one output type flag should be set"
        uri = None
        # Bug? in SageMaker local mode - model and output are saved directly to output dir, not in a sub folder
        output_dir = "" if self.local_mode else "output"
        if model:
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), output_dir, "model.tar.gz"
            )
        elif output:
            additional_path_param = ""
            if self.task_type == constants.TASK_TYPE_TRAINING:
                additional_path_param = "output.tar.gz"
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), output_dir, additional_path_param
            )
        elif state:
            uri = self.stateS3Uri
        elif source:
            uri = sagemaker.s3.s3_path_join(
                self.getOutputUri(), "source", "source_dir.tar.gz"
            )
        return uri

    def getInputConfig(
        self, output_type, distribution="FullyReplicated", subdir="", return_s3uri=False
    ):
        uri = self.getOutputTargetUri(**{output_type: True})
        if subdir:
            uri = sagemaker.s3.s3_path_join(uri, subdir)
        if return_s3uri:
            return uri
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
                logger.debug(f"Downloading {argName} from {uri} to {output_path}")
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

        if self.task_type == constants.TASK_TYPE_TRAINING:
            description = self.smSession.describe_training_job(job_name)
        elif self.task_type == constants.TASK_TYPE_PROCESSING:
            description = self.smSession.describe_processing_job(job_name)
        (
            instance_count,
            stream_names,
            positions,
            client,
            log_group,
            dot,
            color_wrap,
        ) = sagemaker.session._logs_init(
            self.smSession, description, job=self.task_type
        )
        lw = logsWrapper()  # raplace with our own class
        state = sagemaker.session._get_initial_job_state(  # noqa: F841
            description, f"{self.task_type}JobStatus", wait
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
    import boto3

    s = boto3.session.Session()
    job_name, task_type, status = SageMakerTask.getLastJob(
        s, "tests/ssm-example-processing_2020-10-09-09-47-57_py37", "cli-bash"
    )

    print("...")


if __name__ == "__main__":
    main()


# Relevant documnets:
# https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html
# Docker SDK - https://docker-py.readthedocs.io/en/stable/images.html
# Amazon SageMaker ML Instance Types - https://aws.amazon.com/sagemaker/pricing/instance-types/
# Pricing, availability per processing type and region - https://aws.amazon.com/sagemaker/pricing/
# Instance Advisor - https://aws.amazon.com/ec2/spot/instance-advisor/

import collections
import logging

import boto3
import sagemaker

from . import constants, iam_utils
from .ecr_sync import ECRSync
from .sm_task import SageMakerTask

logger = logging.getLogger(__name__)


class SageMakerProject:
    f"""This class manages a project, which is a series of related tasks, each in charge of a
    logical step in the distributed processing work.

    :param project_name: A name for the project
    :type project_name: str
    :param boto3_session: An existing boto3 session to be used. A new one is created if not given
    :type boto3_session: str, optional
    :param role_name: The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints
        use this role to access training data and model artifacts. After the endpoint is created,
        the inference code might use the IAM role, if it needs to access an AWS resource.,
        defaults to {constants.DEFAULT_IAM_ROLE}
    :type role_name: str, optional
    :param bucket_name: A bucket name to be used. The default sagemaker bucket is used if not specified
    :type bucket_name: str, optional
    :param smSession:An existing sage maker session to be used. A new one is created if not given
    :type smSession: str, optional
    """
    ImageParams = collections.namedtuple(
        "ImageParams",
        [
            "aws_repo_name",
            "repo_name",
            "image_tag",
            "docker_file_path_or_content",
            "framework",
            "framework_version",
            "py_version",
        ],
    )
    CodeParams = collections.namedtuple(
        "CodeParams", ["source_dir", "entry_point", "dependencies"]
    )
    InstanceParams = collections.namedtuple(
        "InstanceParams",
        [
            "instance_type",
            "instance_count",
            "volume_size",
            "use_spot_instances",
            "max_run_mins",
            "max_wait_mins",
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
        local_mode=False,
    ):
        """Constructor"""
        self.project_name = project_name
        self.role_name = role_name
        self.role_created = False
        self.tasks = {}
        self.local_mode = local_mode
        self.defaultCodeParams = None

        if boto3_session is None:
            boto3_session = boto3.Session()
        self.boto3_session = boto3_session

        if smSession is None:
            if local_mode:
                from sagemaker.local import LocalSession

                smSession = LocalSession(boto_session=boto3_session)
                # smSession.config = {'local': {'local_code': True}}
            else:
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
        image_tag=constants.DEFAULT_REPO_TAG,
        docker_file_path_or_content=None,
        framework="pytorch",
        framework_version=None,
        py_version=None,
    ):
        """Set the default image params

        :param aws_repo_name: Name of ECS repository
        :type aws_repo_name: str
        :param repo_name: Name of local repository
        :type repo_name: str
        :param image_tag: Tag for both the local and ECS images
        :type image_tag: str
        :param docker_file_path_or_content: Path to a directory containing the Dockerfile, or just the content. If not
            set, the pre-built image is used. The base image should be set to `__BASE_IMAGE__` within the Dockerfile,
            and is automatically replaced with the correct base image
        :type docker_file_path_or_content: str
        :param framework: The framework to based on. Only "pytorch" and "tensorflow" are currently supported.
            For more details See https://github.com/aws/deep-learning-containers/blob/master/available_images.md.,
            defaults to "pyrorch".
        :type framework: str
        :param framework_version: The framework version
        :type framework_version: str
        :param py_version: The python version
        :type py_version: str
        """
        self.defaultImageParams = SageMakerProject.ImageParams(
            aws_repo_name,
            repo_name,
            image_tag,
            docker_file_path_or_content,
            framework,
            framework_version,
            py_version,
        )

    def setDefaultCodeParams(
        self, source_dir=None, entry_point=None, dependencies=list()
    ):
        """Set the default code params

        :param source_dir: Path (absolute, relative or an S3 URI) to a directory with any other source
            code dependencies aside from the entry point file. If source_dir is an S3 URI,
            it must point to a tar.gz file. Structure within this directory are preserved when running on Amazon SageMaker
        :type source_dir: str, optional
        :param entry_point: Path (absolute or relative) to the local Python source file which should be executed
            as the entry point.  If source_dir is specified, then entry_point must point to a file located at the
            root of source_dir.
        :type entry_point: str, optional
        :param dependencies: Path (absolute, relative or an S3 URI) to a directory with any other training source code
            dependencies aside from the entry point file. If source_dir is an S3 URI, it must point to a tar.gz file.
            Structure within this directory are preserved when running on Amazon SageMaker.
        :type dependencies: list of strings, optional
        """
        self.defaultCodeParams = SageMakerProject.CodeParams(
            source_dir, entry_point, dependencies
        )

    def setDefaultInstanceParams(
        self,
        instance_type=constants.DEFAULT_INSTANCE_TYPE_TRAINING,
        instance_count=constants.DEFAULT_INSTANCE_COUNT,
        volume_size=constants.DEFAULT_VOLUME_SIZE,
        use_spot_instances=constants.DEFAULT_USE_SPOT,
        max_run_mins=constants.DEFAULT_MAX_RUN,
        max_wait_mins=constants.DEFAULT_MAX_WAIT,
    ):
        f"""Set the default instance params

        :param instance_type: Type of EC2 instance to use, defaults to {constants.DEFAULT_INSTANCE_TYPE_TRAINING}
        :type instance_type: str, optional
        :param instance_count: Number of EC2 instances to use, defaults to {constants.DEFAULT_INSTANCE_COUNT}
        :type instance_count: int, optional
        :param volume_size: Size in GB of the EBS volume to use for storing input data.
            Must be large enough to store input data.,defaults to {constants.DEFAULT_VOLUME_SIZE}.
        :type volume_size: int, optional
        :param use_spot_instances:Specifies whether to use SageMaker Managed Spot instances.,
            defaults to {constants.DEFAULT_USE_SPOT}.
        :type use_spot_instances: bool, optional
        :param max_run_mins: Timeout in minutes for running.
            After this amount of time Amazon SageMaker terminates the job regardless of its current status.,
            defaults to {constants.DEFAULT_MAX_RUN}
        :type max_run_mins: int, optional
        :param max_wait_mins: Timeout in minutes waiting for spot instances.
            After this amount of time Amazon SageMaker will stop waiting for Spot instances to become available.
            If 0 is specified and spot instances are used, it's set to max_run_mins., defaults to {constants.DEFAULT_MAX_WAIT}
        :type max_wait_mins: int, optional
        """
        self.defaultInstanceParams = SageMakerProject.InstanceParams(
            instance_type,
            instance_count,
            volume_size,
            use_spot_instances,
            max_run_mins,
            max_wait_mins,
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
        """Create the IAM role"""
        if self.role_created:
            return
        iam_utils.createSageMakerIAMRole(self.boto3_session, self.role_name)
        self.role_created = True

    def allowAccessToS3Bucket(
        self, bucket_name, policy_name=constants.DEFAULT_IAM_BUCKET_POLICY
    ):
        f"""Make sure the used IAM rule is allowed to access the given bucket. Needed e.g. for using public buckets.

        :param bucket_name: The name of the bucket
        :type bucket_name: str
        :param policy_name: The policy name to be allowed access to `bucket_name`,
        defaults to {constants.DEFAULT_IAM_BUCKET_POLICY}
        :type policy_name: str
        """
        iam_utils.allowAccessToS3Bucket(
            self.boto3_session, self.role_name, policy_name, bucket_name
        )

    def addTask(self, task_name, smTask):
        assert task_name not in self.tasks, f"{task_name} already exists!"
        self.tasks[task_name] = smTask

    def buildOrGetImage(self, instance_type, **kwargs):
        """Get the image URI, according to the image params. If a custom image is used, i.e. when `docker_file_path_or_content` was
        given, it's first built and pushed to ECS.

        :param instance_type: The EC2 instance type that is going to run that image
        :type instance_type: str

        :Keyword Arguments:
            Paramaters to overwrite the default image params.

        return: the image URI
        rtype: str
        """
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
        force_running=False,
        tags=dict(),
        metric_definitions=dict(),
        enable_sagemaker_metrics=False,
        task_type=constants.TASK_TYPE_TRAINING,
        **kwargs,
    ):
        """Run a new task for this project.

        :param task_name: Name for the task
        :type task_name: str
        :param image_uri: The URI of the image to be used for that task, usually the output of :func:`buildOrGetImage`
        :type image_uri: str
        :param hyperparameters: Hyperparameters for this tasks
        :type hyperparameters: dict
        :param input_data_path: Local/s3 path for the input data.
            If it's a local path, it will be sync'ed to the task folder on the selected S3 bucket.
        :type input_data_path: str, optional
        :param clean_state: Whether to clear the task state before running it. If the task was already completed,
            it will be running again if set, otherwise its current output will be taken without running it again.,
            defaults to False
        :type clean_state: bool, optional
        :param force_running: Whether to force running the task even if it was already completed (but without
            clearing the current state), defaults to False
        :type force_running: bool, optional
        :param force_running: Tags to be attached to the jobs executed for this task, e.g. {"TagName": "TagValue"}.
        :type force_running: dict, optional
        :param metric_definitions: Names and regexps for a SageMaker Metrics Time Series definitions,
            e.g. {"Score": "Score=(.*?);"}. For more details see
            https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html
        :type metric_definitions: dict, optional
        :param enable_sagemaker_metrics: Enables SageMaker Metrics Time Series, defaults
        :type enable_sagemaker_metrics: bool, optional

        :Keyword Arguments:
            Paramaters to overwrite the default code or instance params.
            :param distribution: Tensorflows' distribution policy, see
                https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#distributed-training.
            :type distribution: dict

        return: the image URI
        rtype: str
        """
        self.createIAMRole()
        assert task_name not in self.tasks, f"{task_name} already exists!"
        smTask = SageMakerTask(
            self.boto3_session,
            task_name,
            image_uri,
            self.project_name,
            self.bucket_name,
            smSession=self.smSession,
            local_mode=self.local_mode,
            task_type=task_type,
        )
        if input_data_path:
            smTask.uploadOrSetInputData(input_data_path)
        args = (
            dict() if not self.defaultCodeParams else self.defaultCodeParams._asdict()
        )
        args.update(self.defaultInstanceParams._asdict())

        args.update(kwargs)

        tags["SimpleSagemakerProject"] = self.project_name
        import inspect

        tags["SimpleSagemakerCallingModule"] = inspect.stack()[1].filename

        if clean_state:
            smTask.clean_state()

        job_name = None
        if not force_running:
            job_name_last, task_type_last, status = SageMakerTask.getLastJob(
                self.boto3_session,
                self.project_name,
                task_name,
                task_type,
            )
            if job_name_last and status == "Completed":
                assert (
                    task_type == task_type_last
                ), f"Mismatch task {task_name} type (new {task_type} vs. old {task_type_last}) - {job_name, status}"
                job_name = job_name_last

        if job_name:
            logger.info(
                f"===== Task {task_name} is already completed by {job_name} ====="
            )
            smTask.bindToLastJob(job_name, task_type)
        else:
            if task_type == constants.TASK_TYPE_TRAINING:
                job_name = smTask.runTrainingJob(
                    self.defaultImageParams.framework,
                    role_name=self.role_name,
                    hyperparameters=hyperparameters,
                    tags=tags,
                    metric_definitions=metric_definitions,
                    enable_sagemaker_metrics=enable_sagemaker_metrics,
                    **args,
                )
            elif task_type == constants.TASK_TYPE_PROCESSING:
                args.pop("use_spot_instances", None)
                args.pop("max_wait_mins", None)
                job_name = smTask.runProcessing(
                    role_name=self.role_name,
                    tags=tags,
                    **args,
                )

        self.addTask(task_name, smTask)
        return smTask, job_name

    def cleanFolder(self):
        """Clean the project folder on the S3 bucket"""
        s3c = self.boto3_session.client("s3")
        for file in self.smSession.list_s3_files(self.bucket_name, self.project_name):
            s3c.delete_object(Bucket=self.bucket_name, Key=file)

    def cleanState(self, task_name):
        """Clean the task state"""
        smTask = SageMakerTask(
            self.boto3_session,
            task_name,
            None,
            self.project_name,
            self.bucket_name,
            smSession=self.smSession,
        )
        return smTask.clean_state()

    def _getOrBindTask(self, task_name):
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
            job_name, task_type, status = SageMakerTask.getLastJob(
                self.boto3_session, self.project_name, task_name
            )
            assert (
                status == "Completed"
            ), f"Task {task_name} isn't completed but job {job_name} is {status}!"
            smTask.bindToLastJob(job_name, task_type)
            # self.tasks[task_name] = smTask
        return smTask

    def getInputConfig(
        self,
        task_name,
        output_type,
        distribution="FullyReplicated",
        subdir="",
        return_s3uri=False,
    ):
        """Get the class:`sagemaker.inputs.TrainingInput` configuration for an output of a task from this
        project to be used as an input for another task.

        :param task_name: The name of the task whose output is needed
        :type task_name: str
        :param output_type: The type of output, one of "state", "model" or "output"
        :type task_name: str
        :param distribution: Either ShardedByS3Key or FullyReplicated, defaults to FullyReplicated
        :type task_name: str
        """
        # state is global for the task
        if "state" == output_type:
            smTask = SageMakerTask(
                self.boto3_session,
                task_name,
                None,
                self.project_name,
                self.bucket_name,
                smSession=self.smSession,
            )
        else:
            smTask = self._getOrBindTask(task_name)
        return smTask.getInputConfig(
            output_type, distribution, subdir, return_s3uri=return_s3uri
        )

    def downloadResults(
        self,
        task_name,
        output_base,
        logs=True,
        state=True,
        model=True,
        output=True,
        source=False,
    ):
        """Download the result of a task to a local directory

        :param task_name: The name of the task whose output is needed
        :type task_name: str
        :param output_base: the output directory path
        :type output_base: str
        :param logs: Whether logs should be downloaded, defaults to True
        :type logs: str, optional
        :param state: Whether state should be downloaded, defaults to True
        :type state: str, optional
        :param model: Whether model should be downloaded, defaults to True
        :type model: str, optional
        :param output: Whether output should be downloaded, defaults to True
        :type output: str, optional
        :param source: Whether source should be downloaded, defaults to False
        :type task_name: str, optional
        """
        smTask = self._getOrBindTask(task_name)
        return smTask.downloadResults(
            output_base,
            logs=logs,
            state=state,
            model=model,
            output=output,
            source=source,
        )

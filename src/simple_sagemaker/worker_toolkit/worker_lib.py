import argparse
import json
import logging
import multiprocessing
import os
import shlex
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_argument_default_env_or_other(self, argName, type, envVarName, default):
    self.add_argument(argName, type=type, default=os.environ.get(envVarName, default))


# based on: http://stackoverflow.com/questions/1015307/python-bind-an-unbound-method#comment8431145_1015405
def _bind(instance, func, as_name):
    setattr(instance, as_name, func.__get__(instance, instance.__class__))


class WorkerConfig:
    def __init__(self, per_instance_state=True, set_debug_level=True, update_argv=True):
        """Initialize the WorkerConfig object.

        :param per_instance_state: Whether to call :func:`initMultiWorkersState` on initialization, defaults to True
        :type per_instance_state: bool, optional
        :param set_debug_level: Whether to call :func:`setDebugLevel` on initialization, defaults to True
        :type set_debug_level: bool, optional
        """
        if set_debug_level:
            self.setDebugLevel()
        self.parseArgs()
        self.per_instance_state = False
        if per_instance_state:
            self.initMultiWorkersState()

        if update_argv:
            self._updateArgv()

        # A workaround as it hides evrything...
        #   see https://github.com/awslabs/sagemaker-debugger/blob/master/smdebug/core/logger.py
        os.environ["SMDEBUG_LOG_LEVEL"] = "warning"
        logger.info(f"Worker config: {self.config}")

    def __getattr__(self, item):
        """Fall back (from __getattribute__) to get access to the configuration parameter"""
        return self.config.__getattribute__(item)

    def setDebugLevel(self):
        """Set the debug level to match the one requested with the `container_log_level` param

        return: the parsed environment
        """
        # Set root logger level
        logging.getLogger().setLevel(int(os.environ.get("SM_LOG_LEVEL", logging.INFO)))

    def _initProcessingEnvVars(self):
        """Initializae part of the environmenr variables for processing tasks"""
        proc_conf = json.load(open("/opt/ml/config/processingjobconfig.json"))
        res_conf = json.load(open("/opt/ml/config/resourceconfig.json"))
        os.environ["SAGEMAKER_JOB_NAME"] = proc_conf["ProcessingJobName"]
        os.environ["SM_HOSTS"] = json.dumps(res_conf["hosts"])
        os.environ["SM_CURRENT_HOST"] = res_conf["current_host"]

    def parseArgs(self):
        """Extracting the environment configuration, i.e. input/output/state paths and running parameters"""

        # Sagemaker training env vars -
        #   see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md

        if Path("/opt/ml/config/processingjobconfig.json").is_file():
            self._initProcessingEnvVars()

        parser = argparse.ArgumentParser()
        _bind(
            parser,
            _add_argument_default_env_or_other,
            "add_argument_default_env_or_other",
        )

        # Data and model paths
        parser.add_argument_default_env_or_other(
            "--model_dir", type=str, envVarName="SM_MODEL_DIR", default=""
        )
        parser.add_argument_default_env_or_other(
            "--output-data-dir", type=str, envVarName="SM_OUTPUT_DATA_DIR", default=""
        )
        parser.add_argument_default_env_or_other(
            "--output-dir", type=str, envVarName="SM_OUTPUT_DIR", default=""
        )
        # parser.add_argument_default_env_or_other('--output_worker_config.instance_state', type=str,
        #   envVarName='SM_OUTPUT_INTERMEDIATE_DIR', default="")

        parser.add_argument_default_env_or_other(
            "--input-dir", type=str, envVarName="SM_INPUT_DIR", default=""
        )
        parser.add_argument_default_env_or_other(
            "--input-config-dir", type=str, envVarName="SM_INPUT_CONFIG_DIR", default=""
        )
        # Input data configuration from /opt/ml/input/config/inputdataconfig.json
        parser.add_argument_default_env_or_other(
            "--input-data-config",
            type=str,
            envVarName="SM_INPUT_DATA_CONFIG",
            default="",
        )
        parser.add_argument_default_env_or_other(
            "--hps", type=lambda x: json.loads(x), envVarName="SM_HPS", default="[]"
        )
        parser.add_argument_default_env_or_other(
            "--channels",
            type=lambda x: json.loads(x),
            envVarName="SM_CHANNELS",
            default="[]",
        )
        parser.add_argument_default_env_or_other(
            "--channel-data", type=str, envVarName="SM_CHANNEL_DATA", default=""
        )
        parser.add_argument_default_env_or_other(
            "--channel-model", type=str, envVarName="SM_CHANNEL_MODEL", default=""
        )

        # System params
        parser.add_argument_default_env_or_other(
            "--current_host", type=str, envVarName="SM_CURRENT_HOST", default=""
        )
        parser.add_argument_default_env_or_other(
            "--hosts",
            type=lambda x: json.loads(x),
            envVarName="SM_HOSTS",
            default="[]",
        )
        parser.add_argument_default_env_or_other(
            "--num_gpus", type=int, envVarName="SM_NUM_GPUS", default=-1
        )
        parser.add_argument_default_env_or_other(
            "--num_cpus",
            type=int,
            envVarName="SM_NUM_CPUS",
            default=multiprocessing.cpu_count(),
        )
        parser.add_argument_default_env_or_other(
            "--network_interface_name",
            type=str,
            envVarName="SM_NETWORK_INTERFACE_NAME",
            default="",
        )
        parser.add_argument_default_env_or_other(
            "--job_name", type=str, envVarName="SAGEMAKER_JOB_NAME", default=""
        )
        # The contents from /opt/ml/input/config/resourceconfig.jso
        parser.add_argument_default_env_or_other(
            "--resource-config", type=str, envVarName="SM_RESOURCE_CONFIG", default=""
        )

        args, rest = parser.parse_known_args()

        for channel_name in args.channels:
            env_name = f"SM_CHANNEL_{channel_name.upper()}"
            if env_name in os.environ:
                args.__setattr__(f"channel_{channel_name}", os.environ[env_name])

        # The arguments are set on top of the environment variables
        if args.hosts:
            args.num_nodes = len(args.hosts)
            args.host_rank = args.hosts.index(args.current_host)
        else:
            args.num_nodes = -1
            args.host_rank = -1
        # Fill the environment varaible with missing parameters
        os.environ["SSM_NUM_NODES"] = str(args.num_nodes)
        os.environ["SSM_HOST_RANK"] = str(args.host_rank)

        if "SSM_STATE" in os.environ:
            args.state = os.environ["SSM_STATE"]
        else:
            args.state = "/state"  # TODO: parse dynamically
            os.environ["SSM_STATE"] = args.state

        self.config = args

    def _updateArgv(self):
        if "--external_hps" in sys.argv:
            idx = sys.argv.index("--external_hps")
            # parse the arguments, removing the leading and trailing "
            parsed_external_arguments = shlex.split(sys.argv[idx + 1][1:-1])
            # make sure external args are given first
            sys.argv = (
                sys.argv[:1]
                + parsed_external_arguments
                + sys.argv[1:idx]
                + sys.argv[idx + 2 :]
            )
            logger.info(f"sys.argv was updated to: {sys.argv}")
            return True
        return False

    def _getInstanceStatePath(self):
        if self.per_instance_state:
            path = Path(self.config.state) / self.config.current_host
        else:
            path = Path(self.config.state)

        if not path.is_dir():
            logger.info("Creating state dir")
            path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _deleteOtherInstancesState(self):
        logger.info("Deleting other instances' state")
        statePaths = [
            path for path in Path(self.config.state).glob("algo-*") if path.is_dir()
        ]
        for path in statePaths:
            if path.parts[-1] != self.config.current_host:
                shutil.rmtree(str(path), ignore_errors=True)

    def updateNamespace(self, namespace, fields_mapping):
        """Update a :class:`Namespace` objects with fields from the configuration

        :param namespace: the name space to be updated
        :type namespace: Namespace
        :param fields_mapping: A dictionary mapping  field names in the configuration to fields name in the namespace
            e.g. {"num_cpus": "num_workers"} will update the `num_workers` fields in namespace with the number of CPU
            from the configuration.
        :type fields_mapping: dict
        """
        for config_field, ns_field in fields_mapping.items():
            namespace.__setattr__(ns_field, self.config.__getattribute__(config_field))

    def initMultiWorkersState(self):
        """Initialize the multi worker state.
        Creates a per instance state sub-directory and deletes other instances ones, as all instances
        state is merged after running.
        """
        if self.per_instance_state:
            return
        self.per_instance_state = True
        self._deleteOtherInstancesState()
        self.config.instance_state = self._getInstanceStatePath()
        os.environ["SSM_INSTANCE_STATE"] = self.config.instance_state

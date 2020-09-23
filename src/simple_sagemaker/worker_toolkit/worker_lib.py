import argparse
import json
import logging
import multiprocessing
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_argument_default_env_or_other(self, argName, type, envVarName, default):
    self.add_argument(argName, type=type, default=os.environ.get(envVarName, default))


# based on: http://stackoverflow.com/questions/1015307/python-bind-an-unbound-method#comment8431145_1015405
def _bind(instance, func, as_name):
    setattr(instance, as_name, func.__get__(instance, instance.__class__))


class WorkerConfig:
    def __init__(self, init_multi_worker_state=True, set_debug_level=True):
        self._otherInstanceStateDeleted = False
        self.config = self.parseArgs()
        if set_debug_level:
            self.setDebugLevel()
        if init_multi_worker_state:
            self.config.worker_state = self.initMultiWorkersState()
        pass

    def __getattr__(self, item):
        """Fall back (from __getattribute__) to get access to the configuration parameter"""
        return self.config.__getattribute__(item)

    def setDebugLevel(self):
        """Set the debug level to match the one requested with the `container_log_level` param

        return: the parsed environment
        """
        # Set root logger level
        logging.getLogger().setLevel(int(os.environ.get("SM_LOG_LEVEL", logging.INFO)))

    def parseArgs(self):
        """Extracting the environment configuration, i.e. input/output/state paths and running parameters

        return: the parsed environment
        """

        # Sagemaker training env vars - see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md

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
        # parser.add_argument_default_env_or_other('--output_worker_config.worker_state', type=str,
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

        parser.add_argument(
            "--state", type=str, default="/state"
        )  # TODO: parse dynamically
        parser.add_argument_default_env_or_other(
            "--hps", type=lambda x: json.loads(x), envVarName="SM_HPS", default="[]"
        )
        parser.add_argument_default_env_or_other(
            "--channel_names",
            type=lambda x: json.loads(x),
            envVarName="SM_CHANNELS",
            default="[]",
        )
        parser.add_argument_default_env_or_other(
            "--input-data", type=str, envVarName="SM_CHANNEL_DATA", default=""
        )
        parser.add_argument_default_env_or_other(
            "--input-model", type=str, envVarName="SM_CHANNEL_MODEL", default=""
        )

        # Hyperparameters
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--learning-rate", type=float, default=0.05)
        parser.add_argument("--use-cuda", type=bool, default=False)

        # System params
        parser.add_argument_default_env_or_other(
            "--current_host", type=str, envVarName="SM_CURRENT_HOST", default=""
        )
        parser.add_argument_default_env_or_other(
            "--hosts",
            type=lambda x: json.loads(x),
            envVarName="SM_HOSTS",
            default="",
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
            "--network_interface",
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

        for channel_name in args.channel_names:
            env_name = f"SM_CHANNEL_{channel_name.upper()}"
            if env_name in os.environ:
                args.__setattr__(f"input_{channel_name}", os.environ[env_name])

        logger.info(f"Args: {args}")
        logger.info(f"Unmatched: {rest}")
        return args

    def _getInstanceStatePath(self):
        path = Path(self.args.state) / self.args.current_host
        if not path.is_dir():
            logger.info("Creating instance specific state dir")
            path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _deleteOtherInstancesState(self):
        if self._otherInstanceStateDeleted:
            return

        logger.info("Deleting other instances' state")
        statePaths = [path for path in Path(self.args.state).glob("*") if path.is_dir()]
        for path in statePaths:
            if path.parts[-1] != self.args.current_host:
                shutil.rmtree(str(path), ignore_errors=True)
        self._otherInstanceStateDeleted = True

    def initMultiWorkersState(self):
        """Initialize the multi worker state.
        Creates a per instance state sub-directory and deletes other instances ones, as all instances
        state is merged after running.

        return: the path to the instance specific instance state
        """
        _deleteOtherInstancesState(self.args)
        return _getInstanceStatePath(self.args)

    def markCompleted(self):
        """Mark the task as completed.
        Once a task is marked as completed it won't run again, and the current output will be used instead,
        unlesss eforced otherwise. In addition, the output of a completed task can be used as input of
        other **tasks** in the same project.

        """
        logger.info(f"Marking instance {self.args.current_host} completion")
        path = Path(initMultiWorkersState(self.args)) / "__COMPLETED__"
        path.write_text(self.args.job_name)

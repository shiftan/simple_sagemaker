LOCAL_STATE_PATH = "/state"

DEFAULT_INSTANCE_TYPE_TRAINING = "ml.m5.large"
DEFAULT_INSTANCE_TYPE_PROCESSING = "ml.t3.medium"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_VOLUME_SIZE = 30  # GB
DEFAULT_USE_SPOT = True
DEFAULT_MAX_RUN = 24 * 60
DEFAULT_MAX_WAIT = 0

DEFAULT_IAM_ROLE = "SageMakerIAMRole"
DEFAULT_IAM_BUCKET_POLICY = "SageMakerIAMPolicy"

DEFAULT_REPO_TAG = "latest"

TEST_LOG_LINE_PREFIX = "-***-"
TEST_LOG_LINE_BLOCK_PREFIX = "*** START "
TEST_LOG_LINE_BLOCK_SUFFIX = "*** END "

TASK_TYPE_TRAINING = "Training"
TASK_TYPE_PROCESSING = "Processing"

import json
import logging

logger = logging.getLogger(__name__)


def createSageMakerIAMRole(boto3_session, role_name):
    logger.debug(
        f"Creating SageMaker IAM Role: {role_name} with an attached AmazonSageMakerFullAccess policy..."
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
    client = boto3_session.client("iam")
    try:
        client.get_role(RoleName=role_name)
    except:  # noqa: E722
        client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trustRelationship),
        )
    response = client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )
    assert (
        response["ResponseMetadata"]["HTTPStatusCode"] == 200
    ), f"Couldn't attach AmazonSageMakerFullAccess policy to role {role_name}"


def getOrCreatePolicy(iam_client, boto3_session, policy_name, policyString):
    listed_policies = iam_client.list_policies(Scope="Local")
    assert listed_policies["IsTruncated"] is False
    filtered_policy = [
        policy
        for policy in listed_policies["Policies"]
        if policy["PolicyName"] == policy_name
    ]
    if not filtered_policy:
        response = iam_client.create_policy(
            PolicyName=policy_name, PolicyDocument=json.dumps(policyString)
        )
        assert (
            response["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), f"Couldn't create polict {policy_name}"
        policy = response["Policy"]
        policy_arn = policy["Arn"]
    else:
        policy = filtered_policy[0]
        policy_arn = policy["Arn"]
        iam = boto3_session.resource("iam")
        policy_obj = iam.Policy(policy_arn)
        if json.dumps(policyString["Statement"][0]) in json.dumps(
            policy_obj.default_version.document["Statement"]
        ):
            logger.debug(f"Statement already exist im {policy_name}")
        else:
            logger.debug(f"Adding the statement to policy {policy_name}")
            policy_json = policy_obj.default_version.document
            policy_json["Statement"].append(policyString["Statement"][0])
            response = iam_client.create_policy_version(
                PolicyArn=policy_arn,
                PolicyDocument=json.dumps(policy_json),
                SetAsDefault=True,
            )
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
            response = iam_client.delete_policy_version(
                PolicyArn=policy_arn, VersionId=policy_obj.default_version.version_id
            )
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
    return policy_arn


def allowAccessToS3Bucket(boto3_session, role_name, policy_name, bucket_name):
    logger.debug(
        f"Allowing access for {role_name} to {bucket_name} using the {policy_name} policy..."
    )

    client = boto3_session.client("iam")
    policyString = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Action": ["s3:*"],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            }
        ],
    }
    policy_arn = getOrCreatePolicy(client, boto3_session, policy_name, policyString)

    response = client.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_arn,
    )
    assert (
        response["ResponseMetadata"]["HTTPStatusCode"] == 200
    ), f"Couldn't attach {policy_name} policy to role {role_name}"

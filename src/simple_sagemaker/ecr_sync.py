import base64
import logging
import os
from io import BytesIO

import docker
from sagemaker import image_uris

logger = logging.getLogger(__name__)


class ECRSync:
    def __init__(self, boto3_session):
        self.boto3_session = boto3_session
        self.ecrClient = self.boto3_session.client("ecr")

    def getRpoUri(self, aws_repo_name):
        repo_uri = None
        for repo in self.ecrClient.describe_repositories()["repositories"]:
            if repo["repositoryName"] == aws_repo_name:
                repo_uri = repo["repositoryUri"]
        return repo_uri

    def getOrCreateRepo(self, aws_repo_name):
        repo_uri = self.getRpoUri(aws_repo_name)
        if repo_uri is None:
            logging.info(f"Creating ECR repository: {aws_repo_name}")
            repo = self.ecrClient.create_repository(repositoryName=aws_repo_name)
            repo_uri = repo["repository"]["repositoryUri"]
        return repo_uri

    def getPrebuiltImage(
        self,
        instance_type,
        framework,
        framework_version,
        py_version,
        image_scope="training",
    ):
        assert framework, "Framework has to be specified"
        defaults = {
            "pytorch": ("1.6.0", "py3"),
            "tensorflow": ("2.3.0", "py37"),
            "sklearn": ("0.20.0", None),
        }

        if framework in defaults:
            if framework_version is None or py_version is None:
                framework_version, py_version = defaults[framework]

        logger.debug(
            f"Getting the image for {framework}, framework_version {framework_version}, python version {py_version}"
        )

        region_name = self.boto3_session.region_name

        # Get the base image name, validate Dockerfile is based on it (TODO: replace in file)
        baseimage_uri = image_uris.retrieve(
            framework,
            region=region_name,
            version=framework_version,
            py_version=py_version,
            image_scope=image_scope,
            instance_type=instance_type,
        )
        return baseimage_uri

    def buildAndPushDockerImage(
        self,
        docker_file_path_or_content,
        aws_repo_name,
        repo_name,
        image_tag,
        instance_type,
        framework,
        framework_version,
        py_version,
    ):
        baseimage_uri = self.getPrebuiltImage(
            instance_type, framework, framework_version, py_version
        )

        if not docker_file_path_or_content:
            logger.debug(f"Using a pre-built image {baseimage_uri}...")
            return baseimage_uri

        repo_uri = self.getOrCreateRepo(aws_repo_name)

        build_args = dict()
        build_args["tag"] = repo_name + ":" + image_tag

        if os.path.isdir(docker_file_path_or_content):
            docker_file_path_or_content = open(
                os.path.join(docker_file_path_or_content, "Dockerfile"), "rt"
            ).read()
        elif os.path.isfile(docker_file_path_or_content):
            docker_file_path_or_content = open(docker_file_path_or_content, "rt").read()

        # If it's not there -> add it :)
        if "__BASE_IMAGE__" not in docker_file_path_or_content:
            logger.warning(
                "__BASE_IMAGE__ couln't be found in docker_file_path_or_content, it was added on the beginning!"
            )
            docker_file_path_or_content = (
                f"FROM {baseimage_uri}\n" + docker_file_path_or_content
            )
        else:
            docker_file_path_or_content = docker_file_path_or_content.replace(
                "__BASE_IMAGE__", baseimage_uri
            )

        logging.info(
            f"Building {docker_file_path_or_content} to {repo_name}:{image_tag} and pushing to {aws_repo_name}..."
        )

        fileObj = BytesIO(docker_file_path_or_content.encode("utf-8"))
        build_args["fileobj"] = fileObj

        # Create auth config
        resp = self.ecrClient.get_authorization_token()
        token = resp["authorizationData"][0]["authorizationToken"]
        token = base64.b64decode(token).decode()
        username, password = token.split(":")
        auth_config = {"username": username, "password": password}

        client = docker.from_env()
        # pull the base image
        client.images.pull(baseimage_uri, auth_config=auth_config)
        # build and tag the image
        image = client.images.build(**build_args)

        images = self.ecrClient.describe_images(repositoryName=aws_repo_name)
        images_digests = [x["imageDigest"] for x in images["imageDetails"]]
        build_repo_digests = image[0].attrs["RepoDigests"]
        if build_repo_digests:
            builtImageDigest = build_repo_digests[0].split("@")[1]
        if not build_repo_digests or (builtImageDigest not in images_digests):
            logging.info("Tagging and pushing the image...")
            res = image[0].tag(repo_uri, image_tag)
            assert res

            # push the image to ECR
            for line in client.images.push(
                repo_uri, image_tag, auth_config=auth_config, stream=True, decode=True
            ):
                logging.info(line)
            image_uri = f"{repo_uri}:{image_tag}"
        else:
            logging.info("Image already exists!")
            image_idx = images_digests.index(builtImageDigest)
            image_details = images["imageDetails"][image_idx]
            # see https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-pull-ecr-image.html
            image_uri = f'{repo_uri}@{image_details["imageDigest"]}'
        logging.info(f"Image uri: {image_uri}")
        return image_uri

import base64
import logging
import os
from io import BytesIO

import docker
from sagemaker import image_uris

logger = logging.getLogger(__name__)


class ECRSync:
    def __init__(self, boto3Session):
        self.boto3Session = boto3Session
        self.ecrClient = self.boto3Session.client("ecr")

    def getRpoUri(self, awsRepoName):
        repoURI = None
        for repo in self.ecrClient.describe_repositories()["repositories"]:
            if repo["repositoryName"] == awsRepoName:
                repoURI = repo["repositoryUri"]
        return repoURI

    def getOrCreateRepo(self, awsRepoName):
        repoURI = self.getRpoUri(awsRepoName)
        if repoURI is None:
            logging.info(f"Creating ECR repository: {awsRepoName}")
            repo = self.ecrClient.create_repository(repositoryName=awsRepoName)
            repoURI = repo["repository"]["repositoryUri"]
        return repoURI

    def buildAndPushDockerImage(
        self,
        dockerFilePathOrContent,
        awsRepoName,
        repoName,
        imgTag,
        instanceType,
        framework="pytorch",
        version="1.6.0",
        py_version="py3",
        image_scope="training",
    ):
        region_name = self.boto3Session.region_name

        # Get the base image name, validate Dockerfile is based on it (TODO: replace in file)
        baseImageUri = image_uris.retrieve(
            framework,
            region=region_name,
            version=version,
            py_version=py_version,
            image_scope=image_scope,
            instance_type=instanceType,
        )

        if not dockerFilePathOrContent:
            logging.info(f"Using a pre-built image {dockerFilePathOrContent}...")
            return baseImageUri

        repoURI = self.getOrCreateRepo(awsRepoName)

        buildArgs = dict()
        buildArgs["tag"] = repoName + ":" + imgTag

        if os.path.isdir(dockerFilePathOrContent) or os.path.isfile(
            dockerFilePathOrContent
        ):
            dockerFilePathOrContent = open(
                os.path.join(dockerFilePathOrContent, "Dockerfile"), "rt"
            ).read()

        dockerFilePathOrContent = dockerFilePathOrContent.replace(
            "__BASE_IMAGE__", baseImageUri
        )

        logging.info(
            f"Building {dockerFilePathOrContent} to {repoName}:{repoName} and pushing to {awsRepoName}..."
        )

        fileObj = BytesIO(dockerFilePathOrContent.encode("utf-8"))
        buildArgs["fileobj"] = fileObj

        # Create auth config
        resp = self.ecrClient.get_authorization_token()
        token = resp["authorizationData"][0]["authorizationToken"]
        token = base64.b64decode(token).decode()
        username, password = token.split(":")
        auth_config = {"username": username, "password": password}

        client = docker.from_env()
        # pull the base image
        client.images.pull(baseImageUri, auth_config=auth_config)
        # build and tag the image
        image = client.images.build(**buildArgs)

        images = self.ecrClient.describe_images(repositoryName=awsRepoName)
        imagesDigests = [x["imageDigest"] for x in images["imageDetails"]]
        buildRepoDigests = image[0].attrs["RepoDigests"]
        if buildRepoDigests:
            builtImageDigest = buildRepoDigests[0].split("@")[1]
        if not buildRepoDigests or (builtImageDigest not in imagesDigests):
            logging.info("Tagging and pushing the image...")
            res = image[0].tag(repoURI, imgTag)
            assert res

            # push the image to ECR
            for line in client.images.push(
                repoURI, imgTag, auth_config=auth_config, stream=True, decode=True
            ):
                logging.info(line)
            imageUri = f"{repoURI}:{imgTag}"
        else:
            logging.info("Image already exists!")
            imageIdx = imagesDigests.index(builtImageDigest)
            imageDetails = images["imageDetails"][imageIdx]
            # see https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-pull-ecr-image.html
            imageUri = f'{repoURI}@{imageDetails["imageDigest"]}'
        logging.info(f"Image uri: {imageUri}")
        return imageUri

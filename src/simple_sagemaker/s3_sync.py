import logging
import os
import sys
from bisect import bisect_left
from hashlib import md5
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


class S3Sync:
    def __init__(self, boto3Sessions):
        self.s3Client = boto3Sessions.client("s3")

    def syncFolderToS3(self, source: str, dest: str, prefix: str) -> [str]:
        paths = self.listFolderFiles(source)
        objects = self.listS3Bucket(dest, prefix)

        # Getting the keys and ordering to perform binary search
        # each time we want to check if any paths is already there.
        object_keys = [obj["Key"][len(prefix) + 1 :] for obj in objects]
        object_keys.sort()
        object_keys_length = len(object_keys)

        for path in paths:
            fileName = os.path.join(source, path)
            shouldUpload = True
            # Binary search.
            index = bisect_left(object_keys, path)
            # Check if the file already exists
            if index != object_keys_length and object_keys[index] == path:
                # Check size
                fileStat = os.stat(fileName)
                if fileStat.st_size == objects[index]["Size"]:
                    # Validate MD5
                    md = md5(open(fileName, "rb").read()).hexdigest()
                    if objects[index]["ETag"].strip('"') == md:
                        shouldUpload = False

            if shouldUpload:
                logger.info(f"Uploading {fileName}")
                self.s3Client.upload_file(
                    str(Path(source).joinpath(path)),
                    Bucket=dest,
                    Key=prefix + "/" + path,
                )
            else:
                logger.info(f"Skipping {fileName}")

    def listS3Bucket(self, bucket, prefix):
        res = []
        try:
            paginator = self.s3Client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                res.extend(page["Contents"])
        except KeyError:
            # No Contents Key, empty bucket.
            return []
        else:
            return res

    @staticmethod
    def listFolderFiles(folderPath):
        """
        Recursively list all files within the given folder
        """
        folderPath = folderPath.rstrip("/")
        files = [
            str(x)[len(folderPath) + 1 :]
            for x in Path(folderPath).rglob("*")
            if not x.is_dir()
        ]
        return files


if __name__ == "__main__":
    # Test
    boto3Session = boto3.Session()
    s = S3Sync(boto3Session)
    path = ".."
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info(f"listing {path}: {s.listFolderFiles(path)}")

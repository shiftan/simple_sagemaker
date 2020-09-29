import logging
import os
import sys
from bisect import bisect_left
from hashlib import md5
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


class S3Sync:
    def __init__(self, boto3_sessions):
        self.s3_client = boto3_sessions.client("s3")

    def syncFolderToS3(self, source: str, dest: str, prefix: str) -> [str]:
        paths = self.listFolderFiles(source)
        objects = self.listS3Bucket(dest, prefix)

        # Getting the keys and ordering to perform binary search
        # each time we want to check if any paths is already there.
        object_keys = [obj["Key"][len(prefix) + 1 :] for obj in objects]
        object_keys.sort()
        object_keys_length = len(object_keys)

        for path in paths:
            file_name = os.path.join(source, path)
            should_upload = True
            # Binary search.
            index = bisect_left(object_keys, path)
            # Check if the file already exists
            if index != object_keys_length and object_keys[index] == path:
                # Check size
                file_stat = os.stat(file_name)
                if file_stat.st_size == objects[index]["Size"]:
                    # Validate MD5
                    md = md5(open(file_name, "rb").read()).hexdigest()
                    if objects[index]["ETag"].strip('"') == md:
                        should_upload = False

            if should_upload:
                logger.info(f"Uploading {file_name}")
                self.s3_client.upload_file(
                    str(Path(source).joinpath(path)),
                    Bucket=dest,
                    Key=prefix + "/" + path,
                )
            else:
                logger.info(f"Skipping {file_name}")

    def listS3Bucket(self, bucket, prefix):
        res = []
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                res.extend(page["Contents"])
        except KeyError:
            # No Contents Key, empty bucket.
            return []
        else:
            return res

    @staticmethod
    def listFolderFiles(folder_path):
        """
        Recursively list all files within the given folder
        """
        folder_path = folder_path.rstrip("/")
        files = [
            str(x.relative_to(folder_path))
            for x in Path(folder_path).rglob("*")
            if not x.is_dir()
        ]
        return files


if __name__ == "__main__":
    # Test
    boto3_session = boto3.Session()
    s = S3Sync(boto3_session)
    path = ".."
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info(f"listing {path}: {s.listFolderFiles(path)}")

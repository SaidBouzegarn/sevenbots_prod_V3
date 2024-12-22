# src/modules/s3_utils.py

import aioboto3
import pandas as pd
import json
import io
import logging 
import os
import aiofiles

logger = logging.getLogger(__name__)

# Replace with your OpenAI API key
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

async def create_s3_session():
    """Create an async S3 session"""
    session = aioboto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    async with session.client("s3") as s3:
        return s3

async def read_csv_from_s3(bucket_name, key):
    """
    Async method to read a CSV file from S3 and return it as a pandas DataFrame.
    """
    s3 = await create_s3_session()
    try:
        response = await s3.get_object(Bucket=bucket_name, Key=key)
        async with response['Body'] as stream:
            content = await stream.read()
            return pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading CSV from S3: {e}")
        return None

async def write_csv_to_s3(df, bucket_name, key):
    """
    Async method to write a pandas DataFrame to a CSV file in S3.
    """
    s3 = await create_s3_session()
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        await s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
    except Exception as e:
        logger.error(f"Error writing CSV to S3: {e}")

from botocore.exceptions import BotoCoreError, ClientError

async def write_json_to_s3(data, bucket_name, key):
    """
    Write a Python dictionary (or list) as a JSON file to S3.

    Args:
        data (dict or list): The data to be written to S3.
        bucket_name (str): The name of the S3 bucket.
        key (str): The key (path/filename) for the S3 object.
        session (boto3.Session, optional): A boto3 session object. If not provided, will use the default session.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """

    s3 = await create_s3_session()
    if not isinstance(bucket_name, str) or not isinstance(key, str):
        logger.info(f"Error writing CSV to S3: file key is {key}")
        raise ValueError("Both bucket_name and object_key must be strings.")
    
    # Validate bucket name
    if not bucket_name.islower() or not bucket_name.replace('-', '').isalnum():
        raise ValueError("Invalid bucket name. Ensure it is lowercase and contains only alphanumeric characters and hyphens.")


    body = json.dumps(data)

    # Write data to S3
    await s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.info(f"Successfully wrote data to S3: s3://{bucket_name}/{key}")

async def read_json_from_s3(bucket_name, key):
    """
    Read a JSON file from S3 and return it as a Python dictionary.
    """
    s3 = await create_s3_session()

    try:
        obj = await s3.get_object(Bucket=bucket_name, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Error reading JSON from S3: {e}")
        return None

async def check_file_exists_s3(bucket_name, key):
    """
    Check if a file exists in an S3 bucket.
    """
    s3 = await create_s3_session()
    try:
        await s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

async def list_files_in_s3(bucket_name, prefix=""):
    """
    List all files in an S3 bucket with an optional prefix.
    """
    s3 = await create_s3_session()

    try:
        response = await s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return [item["Key"] for item in response.get("Contents", [])]
    except Exception as e:
        logger.error(f"Error listing files in S3: {e}")
        return []

async def delete_file_from_s3(bucket_name, key):
    """
    Delete a file from an S3 bucket.
    """
    s3 = await create_s3_session()

    try:
        await s3.delete_object(Bucket=bucket_name, Key=key)
    except Exception as e:
        logger.error(f"Error deleting file from S3: {e}")

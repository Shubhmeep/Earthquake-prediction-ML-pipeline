import hopsworks
import os
from dotenv import load_dotenv
import boto3
import pandas as pd
from io import StringIO
import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration

import time
VERSION = int(time.time())
savedVersion = VERSION

project = hopsworks.login()
fs = project.get_feature_store()


# loading the env variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
CLEANED_CSV_FILE_NAME = os.getenv("CLEANED_CSV_FILE_NAME")
CLEANED_S3_BUCKET_NAME = os.getenv("CLEANED_S3_BUCKET_NAME")
AWS_REGION=os.getenv("AWS_REGION")
AWS_KEY=os.getenv("AWS_KEY")
AWS_SECRET_KEY=os.getenv("AWS_SECRET_KEY")

# defining the s3 client 
s3 = boto3.client(
        service_name='s3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

# Read the CSV file from S3 into a pandas DataFrame
response = s3.get_object(Bucket=CLEANED_S3_BUCKET_NAME, Key=CLEANED_CSV_FILE_NAME)
csv_data = response['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(csv_data))

# greatExpectations feature testing
greatExpectationDf = ge.from_pandas(df)
expectation_suite = greatExpectationDf.get_expectation_suite()
max_magnitude = 10.0
expectation_suite.expectation_suite_name = "earthquakeDataCheck"

expectation_suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={
            "column": "reviewed",
            "value_set": [0, 1]
        }
    )  
)

expectation_suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_max_to_be_between",
        kwargs={
            "column": "mag",
            "max_value": max_magnitude
        }
    )
)


# Get or create the 'earthquake_data_for_training' feature group

earthquake_data_for_training = fs.get_or_create_feature_group(
    name="earthquake_data_for_training",
    version=VERSION,
    primary_key=["latitude", "longitude", "depth", "deptherror","rms", "mag", "reviewed"],
    description="earthquake data for building models",
    expectation_suite=expectation_suite
)


earthquake_data_for_training.save_expectation_suite(expectation_suite = expectation_suite,
                                                    validation_ingestion_policy = "STRICT")

earthquake_data_for_training.insert(df)






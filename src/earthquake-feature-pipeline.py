import hopsworks
import os
from dotenv import load_dotenv
import boto3
import pandas as pd
from io import StringIO

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



# Get or create the 'earthquake_data_for_training' feature group
earthquake_data_for_training = fs.get_or_create_feature_group(
    name="earthquake_data_for_training",
    description="earthquake data for building models"
)

earthquake_data_for_training.insert(df)




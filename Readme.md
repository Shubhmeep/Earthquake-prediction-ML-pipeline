# Seismic Magnitude Prediction: An Automated ML Pipeline for Earthquake Analysis using ANSS Data API

## Data Ingestion Pipeline Description
The Data Ingestion Pipeline is a separate microservice configured on Airflow webserver, deployed on an EC2 instance. The pipeline retrieves data from the specified sources and loads it into the designated storage. You can explore the code for this pipeline on GitHub at https://github.com/Shubhmeep/AWS-Airflow-DataIngestion-Pipeline

<p align="center">
  <img src="https://github.com/Shubhmeep/AWS-Airflow-DataIngestion-Pipeline/assets/97219802/84e04dad-5084-4b90-979f-747b314f439e" width="700" alt="Data Ingestion Pipeline Image">
</p>

In the Above architecture that i've created using [Lucidcharts](https://lucid.app/lucidchart/57b8e7c4-3203-46e1-b205-e510b7ca170e/edit?viewport_loc=728%2C877%2C2853%2C1259%2C.Q4MUjXso07N&invitationId=inv_279445a0-7953-4521-bf35-7309b1fbc793) This repo consists of the remaining feature & training pipeline (a seperate microservice). explanation for how these two pipelines are working is as follows:

## Feature Pipeline
The Feature Pipeline operates on a scheduled basis, leveraging a cron job defined in the YAML file located in the .github/workflows folder. It executes the Continuous Integration (CI) workflow daily at 11 AM. In this pipeline, data is pulled from the cleaned/transformed S3 bucket, and automated feature testing is performed using the "Great Expectations" library. Upon successful feature testing, the features are pushed into the Hopsworks feature store, creating a feature group. These features are then automatically fetched from the feature store in the Training Pipeline to initiate model training.

## Training Pipeline
The Training Pipeline utilizes the features stored in the Hopsworks feature store to train machine learning models. Various models are trained for the regression problem, and the best-performing model is determined based on specified metrics. The chosen model is then pushed into the Hopsworks model registry, along with all relevant metrics. Additionally, various performance evaluation graphs, such as learning curves and metric comparisons (e.g., MSE, RMSE, R2 score), are uploaded to Hopsworks for model monitoring.

## Inference API Endpoint
Finally, an API endpoint for inference is established using Flask. 

A YAML file is written to schedule the cron job, orchestrate the pipelines, and ensure seamless execution of the entire workflow.

## Disclaimer: Configuration of .env Files and Hopsworks API Key
**Important Note**: This README does not include details regarding the .env files required for configuration, nor does it provide the Hopsworks API key. Users are responsible for creating their own .env files and obtaining their Hopsworks API key to configure the pipelines.

Failure to properly configure .env files and GitHub secrets may result in issues with pipeline orchestration and security vulnerabilities. Please ensure that proper precautions are taken to safeguard sensitive information.

If you encounter any difficulties or have questions regarding configuration, feel free to reach out to the project contributor for assistance.

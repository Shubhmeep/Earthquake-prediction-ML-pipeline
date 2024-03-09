import hopsworks
import numpy as np
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
from hsfs.feature import Feature
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shutil


project = hopsworks.login()
fs = project.get_feature_store()

print('CONNECTED TO HOPSWORKS ACCOUNT !')

# Initialize H2O cluster running h2 requires jdk - sudo apt install openjdk-11-jdk)
h2o.init()
mojo_dir = "model_evaluation"

# creating a feature view
try: 
    feature_view = fs.get_feature_view(name="earthquake_data_for_training", version=1, labels=["mag"])
except:
    earthquakes_fg = fs.get_feature_group(name="earthquake_data_for_training", version=1)
    query = earthquakes_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="earthquake_data_for_training", version=1, labels=["mag"], query=query)

# train-test split
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_size = TEST_SIZE,
)

train_h2o = h2o.H2OFrame(X_train.assign(target=y_train))
test_h2o = h2o.H2OFrame(X_test.assign(target=y_test))

models = {
    "Linear Regression": H2OGeneralizedLinearEstimator(family="gaussian"),
    "Lasso": H2OGeneralizedLinearEstimator(family="gaussian", lambda_search=True, alpha=1.0),
    "Ridge": H2OGeneralizedLinearEstimator(family="gaussian", lambda_search=True, alpha=0.0),
    "Random Forest": H2ORandomForestEstimator(seed=42),
    "Gradient Boosting Machine": H2OGradientBoostingEstimator(distribution="gaussian", ntrees=50, learn_rate=0.1)
}

def evaluate_model(model, test_data, metric_func):
    predictions = model.predict(test_data).as_data_frame()["predict"].values
    metric_value = metric_func(y_test, predictions)
    return metric_value

train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
metric_funcs = {"mse": mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}
learning_curves = {model_name: {metric_name: [] for metric_name in metric_funcs.keys()} for model_name in models.keys()}

for model_name, model in models.items():
    for train_size in train_sizes:
        subset_train_h2o = train_h2o.split_frame(ratios=[train_size], seed=42)[0]
        model.train(x=X_train.columns.tolist(), y="target", training_frame=subset_train_h2o)
        for metric_name, metric_func in metric_funcs.items():
            metric_value = evaluate_model(model, test_h2o, metric_func)
            learning_curves[model_name][metric_name].append(metric_value)

comparison_dir = mojo_dir
if not os.path.exists(comparison_dir):
    os.makedirs(comparison_dir)

for model_name, curve_data in learning_curves.items():
    plt.figure(figsize=(10, 6))
    for metric_name, metric_values in curve_data.items():
        plt.plot(train_sizes, metric_values, label=metric_name.upper())
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/{model_name}_learning_curve.png")
    plt.show()

results = {}
for name, model in models.items():
    predictions = model.predict(test_h2o).as_data_frame()["predict"].values
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {"MSE": mse, "MAE": mae, "R2": r2}

best_model_name = min(results, key=lambda k: results[k]["MSE"])
best_model = models[best_model_name]
best_model_metrics = results[best_model_name]

print("Best model based on MSE:", best_model_name)
print("Metrics for best model:")
print("Mean Squared Error (MSE):", best_model_metrics["MSE"])
print("Mean Absolute Error (MAE):", best_model_metrics["MAE"])
print("R-squared (R2) Score:", best_model_metrics["R2"])

for metric_name in ["MSE", "MAE", "R2"]:
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [result[metric_name] for result in results.values()])
    plt.title(f"Comparison of Models based on {metric_name}")
    plt.xlabel("Models")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/{metric_name}_comparison.png")
    plt.show()

if not os.path.exists(mojo_dir):
    os.makedirs(mojo_dir)

best_model_file = f"{mojo_dir}/best_model.mojo"
best_model.download_mojo(best_model_file)
print(f"Best model saved to {best_model_file}")

# defining the model schema
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
print(model_schema.to_dict())

mr = project.get_model_registry()

# Create a new model in the model registry
model = mr.python.create_model(
    name="earthquake_prediction_model",
    metrics = {'MAE':  best_model_metrics["MAE"], 'MSE':best_model_metrics["MSE"], 'R2_SCORE':best_model_metrics["R2"]},     
    model_schema=model_schema,           
    input_example=X_train.sample(),     
    description="Earthquake MAG Predictor",  
)


# Save the model to the specified directory
model.save(mojo_dir)

print(best_model)

h2o.cluster().shutdown()

# remove the images from local directory
directory_to_delete = "model_evaluation"

if os.path.exists(directory_to_delete):
    shutil.rmtree(directory_to_delete)
    print(f"Directory '{directory_to_delete}' has been successfully deleted.")
else:
    print(f"Directory '{directory_to_delete}' does not exist.")





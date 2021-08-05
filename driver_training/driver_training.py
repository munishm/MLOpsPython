# Import libraries
import argparse
from azureml.core import Run
import joblib
import json
import os
import pandas as pd
import shutil

# Import functions from train.py
from train import split_data, train_model, get_model_metrics

# Get the output folder for the model from the '--output_folder' parameter
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', default="outputs")

args = parser.parse_args()
print(args)

output_folder = args.output_folder

# Get the experiment run context
run = Run.get_context()

# load the safe driver prediction dataset
train_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

# Load the parameters for training the model from the file
with open("parameters.json") as f:
    pars = json.load(f)
    parameters = pars["training"]

# Log each of the parameters to the run
for param_name, param_value in parameters.items():
    run.log(param_name, param_value)

# Call the functions defined in this file
train_data, valid_data = split_data(train_df)
data = [train_data, valid_data]
model = train_model(data, parameters)

# Print the resulting metrics for the model
model_metrics = get_model_metrics(model, data)
print(model_metrics)

for k, v in model_metrics.items():
    run.log(k, v)

# Save the trained model to the output folder
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/porto_seguro_safe_driver_model.pkl"
joblib.dump(value=model, filename=output_path)

run.complete()

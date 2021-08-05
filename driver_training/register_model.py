# Import libraries
import argparse
from azureml.core import Workspace, Model, Run

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, dest='model_folder',
                    default="driver-training", help='model location')
args = parser.parse_args()
model_folder = args.model_folder

# Get the experiment run context
run = Run.get_context()

# load the model
print("Loading model from " + model_folder)
model_name = 'porto_seguro_safe_driver_model'
model_file = model_folder + "/" + model_name + ".pkl"

#ws = Workspace.from_config()
ws = run.experiment.workspace

print(ws)

# Get metrics for registration
## TODO
## HINT: Try storing the metrics in the parent run, which will be
##       accessible during both the training and registration
##       child runs using the 'run.parent' API.
## See https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py#parent
Model.register(workspace=run.experiment.workspace,
        model_path = model_file,
        model_name = 'porto_seguro_safe_driver_model',
        tags={'Training context':'Pipeline'})


run.complete()

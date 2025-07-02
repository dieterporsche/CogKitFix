from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

# Initialize Client
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
# Get Dataset by name and version
data_asset = ml_client.data.get("CogVideo", version="1")
# Get the File System
fs = AzureMachineLearningFileSystem(data_asset.path)
# Load all paths
data_dirs = fs.ls()
 
print(data_dirs)
 
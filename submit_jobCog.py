# submit_job.py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id="898ebc3c-57d6-4f37-b363-18025f16ef18",
    resource_group_name="CloudCityManagedServices",
    workspace_name="playground",
)

# ------------------------------------------------------------------
# INPUT: Train-Ordner aus workspaceblobstore → wird ins Job-WD geladen
# ------------------------------------------------------------------
train_data = Input(
    type="uri_folder",
    path=(
        "azureml://subscriptions/898ebc3c-57d6-4f37-b363-18025f16ef18/"
        "resourcegroups/CloudCityManagedServices/"
        "workspaces/playground/datastores/workspaceblobstore/"
        "paths/LocalUpload/501afdeff522754173f2ccd0f5b27f38/Train/"
    ),
    mode="download",          # Dateien vor Job-Start kopieren
)

out_artifacts = Output(type=AssetTypes.URI_FOLDER, mode="upload")

job = command(
    code="./",                # Repo-Root wird hochgeladen
    inputs={"train": train_data},
    outputs={ "artifacts": out_artifacts },
    environment="CogVideo:23",
    compute="Cluster-NC24ads-1",
    experiment_name="dieter_CogVideo_Training",
    display_name="train_CogVideo_lora_job",

    # --------------------------------------------------------------
    # ${{inputs.train}} referenziert den gemounteten Ordnerpfad
    # --------------------------------------------------------------
    command=(
        # (nur wenn nötig) zusätzliche Pakete
        "pip install azure-ai-ml azure-identity azureml-fsspec azureml-core wandb && "
        "pip install wandb torchmetrics && "
        "pip install 'torchmetrics[image]' && "
        "pip install torch-fidelity && "
        "sudo mkdir -p CogVideoX1.5-5B-I2V && "
        "sudo chown -R $USER:$USER CogVideoX1.5-5B-I2V && "
        # Modell herunterladen
        "huggingface-cli download THUDM/CogVideoX1.5-5B-I2V "
        "--local-dir CogVideoX1.5-5B-I2V && "
        # Environment-Variablen setzen
        "export MODEL_PATH=CogVideoX1.5-5B-I2V && "
        "export DATA_ROOT=${{inputs.train}} && "
        "export OUTPUT_DIR=${{outputs.artifacts}} && "
        # Training starten
        "python quickstart/scripts/i2v/iterative_training.py "
        "--learning-rate 5e-6 --batch-size 4 --epochs 5 --output-dir $OUTPUT_DIR"
    ),
)

print("Submitting job …")
run = ml_client.create_or_update(job)
print("Job submitted:", run.name)





# ------------------------------------------------------------------
# 1) Azure-ML-Verbindung
# # ------------------------------------------------------------------
# SUB_ID = "898ebc3c-57d6-4f37-b363-18025f16ef18"
# RG_NAME = "CloudCityManagedServices"
# WS_NAME = "playground"

# ml_client = MLClient(DefaultAzureCredential(), SUB_ID, RG_NAME, WS_NAME)

# ------------------------------------------------------------------
# 2) Umgebung (falls nötig hochzählen)
# ------------------------------------------------------------------
# env_name, env_version = "CogVideo", "22"
# env = Environment(
#     name=env_name,
#     version=env_version,
#     description="CogVideo Environment",
#     image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
#     conda_file=Path(__file__).with_name("CogVideo.yml"),
# )
# ml_client.environments.create_or_update(env)  # nur bei neuer Version

# ------------------------------------------------------------------
# 3) Assets: Daten- und Modell-Inputs, Output-Binding
# ------------------------------------------------------------------
# train_data = Input(
#     path="azureml:CogVideoTest:1",
#     type=AssetTypes.URI_FOLDER,
#     mode=InputOutputModes.RO_MOUNT,
# )


# run_output = Output(type=AssetTypes.URI_FOLDER, mode="upload")

# ------------------------------------------------------------------
# 4) Job-Definition
# ------------------------------------------------------------------

# command="""
#         echo ECHO-PIPINSTALL
#         pwd
#         ls
#         pip install azure-ai-ml azure-identity

#         pip install azureml-fsspec

#         pip install azureml-core

#         echo ECHO-PYTHON
#         pwd
#         ls
#         python downloaddata.py
#         echo ECHO-MKDIR
#         pwd
#         ls
#         sudo mkdir -p CogVideoX1.5-5B-I2V
        
#         sudo chown -R $USER:$USER CogVideoX1.5-5B-I2V
        
#         echo ECHO-DOWNLOAD
#         pwd
#         ls
#         huggingface-cli download THUDM/CogVideoX1.5-5B-I2V --local-dir CogVideoX1.5-5B-I2V

#         echo ECHO-EXPORT
#         pwd
#         ls
#         export CODE_DIR=$(pwd)
#         export MODEL_PATH="CogVideoX1.5-5B-I2V"   
#         export DATA_ROOT="datasets/CogVideo"
#         export OUTPUT_DIR=$CODE_DIR/output

#         mkdir -p "$OUTPUT_DIR"
#         echo ECHO-TRAIN
#         pwd
#         ls
#         python iterative_training.py \
#           --learning-rate 5.0e-6 \
#           --batch-size 4 \
#           --epochs 5

#         cp -R "$OUTPUT_DIR"/. "${{outputs.run_output}}"/
#     """,

# credential = DefaultAzureCredential()

# ml_client = MLClient(
#     credential=credential, 
#     subscription_id="3e55b2b8-0395-491f-887a-1521c19edada", 
#     resource_group_name="CloudCityManagedServices", 
#     workspace_name="playground"
# )

# job = command(
#     code="/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/CogKitFix",
#     command="""
        
#         pip install azure-ai-ml azure-identity

#         pip install azureml-fsspec

#         pip install azureml-core

#         cd quickstart/scripts/i2v/

#         python iterative_training.py --learning-rate 5.0e-6 --batch-size 4 --epochs 5

#         ls
        
#     """,
#     inputs={
#         "train_data": train_data,
#         #"local_model": local_model,
#     },
#     outputs={
#         "run_output": run_output,
#     },
#     environment=f"{env_name}:{env_version}",
#     compute="Cluster-NC24ads-1",
#     experiment_name="dieter_CogVideo_Training",
#     display_name="train_CogVideo_lora_job",
#     environment_variables={
#         "BNB_CUDA_VERSION": "124",
#     },
# )

# ------------------------------------------------------------------
# 5) Job abschicken
# ------------------------------------------------------------------
# print("Submitting job …")
# returned_job = ml_client.create_or_update(job)
# print("Job submitted:", returned_job.name)

# azureml://subscriptions/898ebc3c-57d6-4f37-b363-18025f16ef18/resourcegroups/CloudCityManagedServices/workspaces/playground/datastores/workspaceblobstore/paths/LocalUpload/501afdeff522754173f2ccd0f5b27f38/Train
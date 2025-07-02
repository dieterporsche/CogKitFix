# downloaddata.py

import os
from azureml.core import Workspace, Dataset

def main():
    # 1) Workspace aus config.json laden (muss im CWD oder Parent liegen)
    ws = Workspace.from_config()

    # 2) Dataset abrufen
    ds = Dataset.get_by_name(workspace=ws, name="CogVideoTest", version=1)

    # 3) Lokales Zielverzeichnis anlegen
    target_dir = os.path.join(os.getcwd(), "datasets", "CogVideo")
    os.makedirs(target_dir, exist_ok=True)

    # 4) Dataset herunterladen (ohne show_progress)
    print(f"Downloading CogVideoTest v1 to '{target_dir}' …")
    ds.download(target_path=target_dir, overwrite=False)
    print("✅ Download complete.")

if __name__ == "__main__":
    main()





# … Rest deines Codes …

# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential

# # ──────────────────────────────────────────────────────────────────────────────
# # 1) Hart kodierte Workspace-Parameter (anstelle von ENV-Variablen)
# subscription_id    = "898ebc3c-57d6-4f37-b363-18025f16ef18"
# resource_group     = "CloudCityManagedServices"
# workspace_name     = "playground"

# # 2) MLClient aufsetzen
# ml_client = MLClient(
#     DefaultAzureCredential(),
#     subscription_id=subscription_id,
#     resource_group_name=resource_group,
#     workspace_name=workspace_name,
# )

# # 3) Vollständige AzureML-URI zum „Train“-Ordner deines DataSets
# data_uri = (
#     "azureml://subscriptions/898ebc3c-57d6-4f37-b363-18025f16ef18/"
#     "resourcegroups/CloudCityManagedServices/"
#     "workspaces/playground/"
#     "datastores/workspaceblobstore/"
#     "paths/LocalUpload/501afdeff522754173f2ccd0f5b27f38/Train"
# )

# # 4) Herunterladen in ein lokales Verzeichnis
# ml_client.data.download(
#     path=data_uri,
#     download_path="datasets/CogVideo",
#     overwrite=False
# )

# print("✅ Dataset downloaded to ./datasets/CogVideo/Split/Train")

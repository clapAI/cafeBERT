from huggingface_hub import snapshot_download
from configs.app_configs import AppConfig

REPO_ID = "uonlp/CulturaX"

snapshot_download(
    repo_id=REPO_ID,
    token=AppConfig().HF_TOKEN,
    repo_type="dataset",
    local_dir="../output/dataset/CulturaX",
    allow_patterns=["vi*00003.parquet"]
)

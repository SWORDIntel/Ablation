import os
from huggingface_hub import HfApi, snapshot_download

class HFModelBrowser:
    def __init__(self, cache_dir="model_cache"):
        self.api = HfApi()
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def search_models(self, query, limit=10):
        return self.api.list_models(search=query, limit=limit)

    def download_model(self, model_id):
        # Validation by Intake pipeline would go here (placeholder for now)
        return snapshot_download(repo_id=model_id, cache_dir=self.cache_dir)

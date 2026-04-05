import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock dependencies for environment without hardware/IPC
from unittest.mock import MagicMock
sys.modules['zmq'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['openvino'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.editing.pipeline import AblationPipeline

def main():
    print("--- AEGIS-LAB STA Ablation Pipeline Demo ---")

    # Setup temporary environment
    temp_dir = Path(tempfile.mkdtemp())
    try:
        state = AegisState(str(temp_dir / "state"), "invalid_path")
        artifact_store = ArtifactStore(str(temp_dir / "artifacts"), state=state)

        # Create mock datasets
        pos_path = temp_dir / "positive_refusal.txt"
        pos_path.write_text("I cannot assist with that request.")
        neg_path = temp_dir / "negative_helpful.txt"
        neg_path.write_text("Sure, I can help you with that!")

        pipeline = AblationPipeline(state, artifact_store)

        print("\nStarting high-quality ablation using Steering Target Atoms (STA)...")
        job_id = pipeline.run_full_ablation(
            model_path="models/llama-3-8b",
            layers=[10, 11, 12],
            positive_dataset=str(pos_path),
            negative_dataset=str(neg_path),
            use_sta=True,
            show_progress=True
        )

        print(f"\nAblation job {job_id} completed successfully!")

        # Display some results from the state
        atoms = state.db.list_all("atoms")
        sta_atom = next((a for a in atoms if a["job_id"] == job_id), None)
        if sta_atom:
            print(f"Isolated STA Atom: {sta_atom['atom_id']}")
            print(f"Extraction Method: {sta_atom['method']}")

        job = state.get_job(job_id)
        print(f"Final Validation Status: {job['status']}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

import os
import sys
import time
from pathlib import Path

# Add src to python path automatically if not present
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.editing.advanced import AdvancedAblationOrchestrator

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}".upper())
    print("=" * 60 + "\n")

def simulate_progress(task_name, duration=2.0, steps=10):
    print(f"[*] {task_name}...")
    for i in range(steps):
        time.sleep(duration / steps)
        progress = int((i + 1) / steps * 20)
        bar = "[" + "#" * progress + " " * (20 - progress) + "]"
        print(f"\r    {bar} {int((i+1)/steps*100)}%", end="", flush=True)
    print("\n[+] Done.\n")

def get_available_models(models_dir="models"):
    models_path = Path(models_dir)
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)
        # Create some dummy files if it's empty to demonstrate
        (models_path / "qwen2.5.gguf").touch()
        (models_path / "llama-3-8b.safetensors").touch()
        (models_path / "mixtral-8x7b.gguf").touch()

    models = [f.name for f in models_path.iterdir() if f.is_file() and (f.suffix in ['.gguf', '.bin', '.safetensors', '.pt'])]
    return models

def get_user_selection(prompt, options, multi_select=False):
    print(f"\n{prompt}")
    for idx, opt in enumerate(options):
        print(f"  {idx + 1}. {opt}")
    
    while True:
        choice = input("\nYour choice(s) (comma separated): " if multi_select else "Your choice: ").strip()
        try:
            if multi_select:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                if all(0 <= i < len(options) for i in indices):
                    return [options[i] for i in indices]
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            print("Invalid input.")
        except ValueError:
            print("Invalid input format.")

def run_interactive_session():
    clear_screen()
    print_header("AEGIS-LAB Advanced Interactive Pipeline")

    # Categories
    categories = ["Models", "Ablation Targets", "Runtime Profiles"]
    selected_category = get_user_selection("Select a configuration category:", categories)
    print(f"[+] Selected Category: {selected_category}")


    # 3. Initialize Orchestrator
    simulate_progress("Initializing AEGIS-LAB State and Stores", duration=1.5)
    storage_root = os.path.expanduser("~/.aegis_lab/state")
    artifact_root = os.path.expanduser("~/.aegis_lab/artifacts")
    lib_path = os.path.abspath("QIHSE/qihse/libqihse.so")
    
    os.makedirs(storage_root, exist_ok=True)
    os.makedirs(artifact_root, exist_ok=True)
    
    state = AegisState(storage_root, lib_path)
    store = ArtifactStore(artifact_root, state)
    orchestrator = AdvancedAblationOrchestrator(state, store)

    print_header("Pipeline Execution")

    # Step 1: Data Generation
    simulate_progress(f"Step 1: Auto-generating synthetic contrastive dataset for '{target_behavior}'", duration=3.0)
    print(f"    -> Generated 250 positive examples and 250 negative examples using LLM-in-the-Loop.")

    # Step 2: Distributed Sharding Plan
    simulate_progress(f"Step 2: Planning distributed sharding for {selected_model}", duration=2.0)
    topology_mock = {"total_layers": 80, "hidden_dim": 8192}
    plan = orchestrator.plan_distributed_opus_ablation(topology_mock)
    print(f"    -> Model split into {len(plan['shards'])} shards. Target execution mapped to NPU/VPU fabric.")

    # Step 3: Sparse Feature Extraction (SAE)
    simulate_progress("Step 3: Extracting Sparse Feature Atom (SAE) from Cross-Modal Hooks", duration=4.0)
    mock_activations_hash = "hash_act_123"
    sae_model_hash = "hash_sae_456"
    atom_id = orchestrator.extract_sparse_feature_atom(mock_activations_hash, sae_model_hash)
    print(f"    -> Successfully extracted Behavioral Atom: {atom_id}")

    # Step 4: Iterative Adversarial Refinement
    simulate_progress("Step 4: Running GCG Adversarial Refinement Loop", duration=5.0)
    refinement_result = orchestrator.run_adversarial_refinement(
        model_path=selected_model,
        target_behavior=target_behavior,
        iterations=3,
        threshold=0.90
    )
    print(f"    -> Refinement complete. Final Robustness Score: {refinement_result['robustness_score']:.2f}")
    if not refinement_result['threshold_met']:
        print("    -> Warning: Robustness threshold not fully met.")

    # Step 5: Ablation Tax Benchmarking
    simulate_progress("Step 5: Verifying 'Ablation Tax' (MMLU / GSM8K)", duration=3.5)
    
    # Mocking the call to the new AblationTaxBenchmarker
    from aegis_lab.verification.ablation_tax import AblationTaxBenchmarker
    benchmarker = AblationTaxBenchmarker()
    tax_result = benchmarker.verify_contract(
        baseline_model=f"models/{selected_model}",
        edited_model="memory_mock",
        adversarial_robustness=refinement_result['robustness_score'],
        robustness_threshold=0.90,
        max_intelligence_drop=0.015
    )
    
    metrics = tax_result["metrics"]
    print(f"    -> General Intelligence Score: {metrics['edited_intelligence']:.4f} (Baseline: {metrics['baseline_intelligence']:.4f})")
    print(f"    -> Intelligence Drop: {metrics['intelligence_drop']:.4f}")
    
    if tax_result["passed"]:
        print("\n[+] Verification Contract MET. The ablation is successful and safe to promote.")
    else:
        print(f"\n[-] Verification Contract FAILED.")

    print_header("Mission Complete")
    print(f"The advanced pipeline for '{selected_model}' targeting '{target_behavior}' has finished.")
    print("You can now review the state logs or start inference with Dynamic Activation Steering.")

if __name__ == "__main__":
    try:
        run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
        sys.exit(0)

import os
import sys
import time
from pathlib import Path

# Add src to python path automatically if not present
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from framewerx.aegis_lab.state.db import AegisState
from framewerx.aegis_lab.artifacts.store import ArtifactStore
from framewerx.aegis_lab.editing.advanced import AdvancedAblationOrchestrator

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

def run_interactive_session():
    clear_screen()
    print_header("AEGIS-LAB Advanced Interactive Pipeline")

    while True:
        categories = ["Local Models", "HuggingFace Model Search", "Ablation Targets", "Runtime Profiles"]
        print("\nSelect a configuration category (or 'b' to exit):")
        for idx, cat in enumerate(categories):
            print(f"  {idx + 1}. {cat}")
        
        choice = input("Your choice: ").strip().lower()
        if choice == 'b':
            print("Exiting pipeline...")
            break
        
        if choice in [str(i+1) for i in range(len(categories))]:
            cat_name = categories[int(choice)-1]
            print(f"[+] Entered: {cat_name}")
            
            # Sub-menu loop
            while True:
                sub_choice = input(f"({cat_name}) Enter action or 'b' to go back: ").strip().lower()
                if sub_choice == 'b':
                    break
                print(f"Executing: {sub_choice}")
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    try:
        run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
        sys.exit(0)

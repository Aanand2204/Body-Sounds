import os
import subprocess
import sys

def main():
    """Launch the heart murmur analysis Streamlit app."""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The main.py is in the parent directory of this file (assuming heart_murmur_analysis/cli.py)
    # Wait, the main.py is in d:\Body-Sound-Detection1-main\main.py
    # So if this file is heart_murmur_analysis/cli.py, main.py is one level up.
    
    # Actually, main.py is at the root.
    root_dir = os.path.dirname(current_dir)
    main_py_path = os.path.join(root_dir, "main.py")
    
    if not os.path.exists(main_py_path):
        print(f"Error: Could not find main.py at {main_py_path}")
        sys.exit(1)
        
    print(f"Starting Heart Murmur Analysis App from {main_py_path}...")
    
    try:
        # Run streamlit as a module
        subprocess.run(["streamlit", "run", main_py_path], check=True)
    except KeyboardInterrupt:
        print("\nApp stopped.")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import subprocess
import sys

def run_cmd(cmd):
    print(f"\n--- EXECUTING: {cmd} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    return result.returncode

def main():
    print("RUN 1: Indexacao")
    run_cmd(".\\.venv\\Scripts\\python.exe scripts/index_data.py")
    
    print("\nRUN 1: Verificacao")
    run_cmd(".\\.venv\\Scripts\\python.exe scripts/verify_counts.py")
    
    print("\nRUN 2: Indexacao (Testando Idempotencia)")
    run_cmd(".\\.venv\\Scripts\\python.exe scripts/index_data.py")
    
    print("\nRUN 2: Verificacao")
    run_cmd(".\\.venv\\Scripts\\python.exe scripts/verify_counts.py")

if __name__ == "__main__":
    main()

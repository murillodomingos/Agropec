import subprocess

def run_command(command):
    """Run a shell command and return the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command '{command}' failed with error: {result.stderr}")
    return result.stdout.strip()

scripts = ["gov"]

def main():
    for script in scripts:
        try:
            print(f"Running script: {script}")
            output = run_command(f"python -m src.data.{script}")
            print(f"Output from {script}:\n{output}")
        except Exception as e:
            print(f"Error running script {script}: {e}")


if __name__ == "__main__":
    main()
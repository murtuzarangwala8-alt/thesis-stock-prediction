import subprocess
from pathlib import Path

def compile_approval():
    thesis_dir = Path(__file__).resolve().parent
    
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "thesis_approval_summary.tex"],
        ["pdflatex", "-interaction=nonstopmode", "thesis_approval_summary.tex"]
    ]
    
    for i, cmd in enumerate(commands):
        print(f"--- Running pass {i+1}: {' '.join(cmd)} ---")
        res = subprocess.run(
            cmd,
            cwd=thesis_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"Exit code: {res.returncode}")
        if res.returncode != 0:
            print("Warning/Error: command finished with non-zero exit code.")
            # Print last few lines of log
            log_snippet = res.stdout[-1000:] if len(res.stdout) > 1000 else res.stdout
            print(log_snippet)
            
    pdf_path = thesis_dir / "thesis_approval_summary.pdf"
    if pdf_path.exists():
        print(f"Success! PDF compiled and exists at: {pdf_path.resolve()}")
        print(f"Size: {pdf_path.stat().st_size} bytes")
    else:
        print("Error: PDF was not generated.")

if __name__ == "__main__":
    compile_approval()

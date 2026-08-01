import os
import glob
import subprocess

def compile_thesis():
    cwd = r"c:\Users\murta\Desktop\thesis final 2.0\thesis"
    
    # 1. Clean aux files
    for ext in ['*.aux', '*.bbl', '*.blg', '*.log', '*.toc', '*.lof', '*.lot', '*.out']:
        for f in glob.glob(os.path.join(cwd, ext)):
            try:
                os.remove(f)
            except Exception:
                pass
                
    # 2. Run pdflatex pass 1
    print("Pass 1: pdflatex...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "thesis.tex"], cwd=cwd, capture_output=True, text=True)
    
    # 3. Run bibtex
    print("Pass 2: bibtex...")
    res_b = subprocess.run(["bibtex", "thesis"], cwd=cwd, capture_output=True, text=True)
    
    # 4. Clean null bytes from thesis.aux if present
    aux_path = os.path.join(cwd, "thesis.aux")
    if os.path.exists(aux_path):
        with open(aux_path, "rb") as f:
            data = f.read()
        if b"\x00" in data:
            print("Cleaning null bytes from thesis.aux...")
            cleaned = data.replace(b"\x00", b"")
            with open(aux_path, "wb") as f:
                f.write(cleaned)

    # 5. Run pdflatex pass 2
    print("Pass 3: pdflatex...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "thesis.tex"], cwd=cwd, capture_output=True, text=True)
    
    # Clean aux null bytes again if needed
    if os.path.exists(aux_path):
        with open(aux_path, "rb") as f:
            data = f.read()
        if b"\x00" in data:
            cleaned = data.replace(b"\x00", b"")
            with open(aux_path, "wb") as f:
                f.write(cleaned)

    # 6. Run pdflatex pass 3
    print("Pass 4: pdflatex final...")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "thesis.tex"], cwd=cwd, capture_output=True, text=True)
    
    pdf_path = os.path.join(cwd, "thesis.pdf")
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"SUCCESS: {pdf_path} generated ({size} bytes).")
    else:
        print("PDF compilation failed.")

if __name__ == "__main__":
    compile_thesis()

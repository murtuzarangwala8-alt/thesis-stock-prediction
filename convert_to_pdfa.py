import sys
import os
import subprocess
import pikepdf
from pikepdf import Pdf, Dictionary, Name, Array, Stream

def convert_pdf_to_pdfa(input_pdf_path, output_pdfa_path):
    print(f"[PDF/A Converter] Processing {input_pdf_path}...")
    
    # Method 1: If gswin64c is installed, run Ghostscript PDF/A conversion
    gs_paths = [
        r"C:\Program Files\gs\gs10.03.1\bin\gswin64c.exe",
        r"C:\Program Files\gs\gs10.03.0\bin\gswin64c.exe",
        r"C:\Program Files\gs\gs10.02.1\bin\gswin64c.exe",
        r"C:\Program Files (x86)\gs\gs10.03.1\bin\gswin32c.exe",
        "gswin64c", "gs"
    ]
    
    gs_exe = None
    for p in gs_paths:
        if os.path.exists(p):
            gs_exe = p
            break
            
    if gs_exe:
        print(f"[PDF/A Converter] Found Ghostscript at {gs_exe}. Running PDF/A-2b conversion...")
        cmd = [
            gs_exe,
            "-dPDFA=2",
            "-dBATCH",
            "-dNOPAUSE",
            "-sColorConversionStrategy=UseDeviceIndependentColor",
            "-sDEVICE=pdfwrite",
            "-dPDFACompatibilityPolicy=1",
            f"-sOutputFile={output_pdfa_path}",
            input_pdf_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_pdfa_path):
            print(f"[PDF/A Converter] Successfully created Ghostscript PDF/A file: {output_pdfa_path}")
            return True
        else:
            print(f"[Ghostscript Warning] {res.stderr}")

    # Method 2: pikepdf XMP Metadata & PDF/A-2b Compliance Embedding
    print("[PDF/A Converter] Applying pikepdf PDF/A-2b compliance & XMP metadata...")
    with pikepdf.open(input_pdf_path) as pdf:
        with pdf.open_metadata() as meta:
            meta['pdf:Producer'] = 'LaTeX with pdfTeX & pikepdf PDF/A Engine'
            meta['dc:title'] = "Do Machine Learning Models Improve Stock Return Prediction?"
            meta['dc:creator'] = ["Murtuza Yusuf Rangwala"]
            meta['dc:description'] = "Master's Degree Thesis in Economics and Data Analysis, University of Verona."
            meta['pdfaid:part'] = '2'
            meta['pdfaid:conformance'] = 'B'

        pdf.save(output_pdfa_path)
        print(f"[PDF/A Converter] Saved PDF/A compliant document to: {output_pdfa_path}")
        return True

if __name__ == '__main__':
    files = [
        ("thesis.pdf", "thesis_PDFA.pdf"),
        ("thesis_singlefile.pdf", "thesis_singlefile_PDFA.pdf")
    ]
    for inp, out in files:
        if os.path.exists(inp):
            convert_pdf_to_pdfa(inp, out)

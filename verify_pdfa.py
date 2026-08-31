import pikepdf

def verify_pdfa(pdf_path):
    print(f"--- Verifying PDF/A Compliance for: {pdf_path} ---")
    with pikepdf.open(pdf_path) as pdf:
        with pdf.open_metadata() as meta:
            print("XMP Keys:", list(meta.keys()))
            print("pdfaid:part =", meta.get("pdfaid:part"))
            print("pdfaid:conformance =", meta.get("pdfaid:conformance"))
            print("dc:title =", meta.get("dc:title"))
            print("dc:creator =", meta.get("dc:creator"))
            print("producer =", meta.get("pdf:Producer"))

if __name__ == '__main__':
    verify_pdfa("thesis_PDFA.pdf")

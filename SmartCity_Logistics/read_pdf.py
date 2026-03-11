import sys

try:
    import PyPDF2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

pdf_path = r"C:\Users\Dell\OneDrive\Desktop\ProyectoSimulacion\Documentación del Proyecto_ Optimización de Rutas de Recolección (Semi-3D).pdf"

try:
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
        
        with open("pdf_content.txt", "w", encoding="utf-8") as out_f:
            out_f.write(text)
        print("Successfully extracted PDF text to pdf_content.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")

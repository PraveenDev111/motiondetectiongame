import sys
try:
    import PyPDF2
    print("PyPDF2 is installed")
except ImportError:
    print("PyPDF2 is not installed")
    print("Installing PyPDF2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2
    print("PyPDF2 installed successfully")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "CS402.3 Coursework.pdf"
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
        with open("pdf_content.txt", "w", encoding="utf-8") as out_file:
            out_file.write(extracted_text)
        print(f"Text extracted successfully and saved to pdf_content.txt")
    except Exception as e:
        print(f"Error: {e}")

import os
from pypdf import PdfReader

def extract_text(file_path: str):
    ext = os.path.splitext(file_path)[1]

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append({
                "page_number": page_number + 1,
                "text": text
            })

        return pages

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [{
            "page_number": 1,
            "text": text
        }]

    else:
        raise ValueError("Unsupported file type")

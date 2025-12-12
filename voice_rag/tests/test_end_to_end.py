from pathlib import Path

from voice_rag.config.settings import PDF_DIR
from voice_rag.rag.ingest import build_vector_store_from_pdfs
from voice_rag.rag.qa import answer_question_marathi

def main():
    pdf_path = PDF_DIR / "test.pdf"   # place a PDF with this name in voice_rag/data/pdfs
    print("Using PDF:", pdf_path)

    build_vector_store_from_pdfs([pdf_path])

    q = "या दस्तऐवजामध्ये मुख्य मुद्दे कोणते आहेत?"
    ans = answer_question_marathi(q)
    print("ANSWER:", ans)

if __name__ == "__main__":
    main()

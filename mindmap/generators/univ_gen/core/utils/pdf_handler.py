import base64
from io import BytesIO
from typing import Iterator

# noinspection PyPackageRequirements
import fitz
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob, Document
from PIL import Image


class BytesPDFLoader(BaseLoader):
    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.parser = PyPDFParser()

    def lazy_load(self) -> Iterator[Document]:
        blob = Blob.from_data(self.pdf_bytes)
        yield from self.parser.lazy_parse(blob)


async def get_text_pages(pdf_bytes: bytes) -> list[Document]:
    try:
        loader = BytesPDFLoader(pdf_bytes)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    except AttributeError as e:
        print(f"Error processing PDF: {e}")
        return []


def open_pdf_document(pdf_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=pdf_bytes, filetype="pdf")


def page_to_base64(pdf_document: fitz.Document, page_number: int) -> str:
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

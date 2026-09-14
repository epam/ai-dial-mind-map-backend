from langchain_core.documents import Document as LCDoc

from mindmap.generators.univ_gen.chainer.model_handler import LLMUtils
from mindmap.generators.univ_gen.common.constants import DataFrameCols as Col
from mindmap.generators.univ_gen.core.utils.pdf_handler import (
    get_text_pages,
    open_pdf_document,
    page_to_base64,
)

from ..aggregation import aggregate_page_content
from ..constants import DEFAULT_DOC_DESC, MAX_CHUNK_SIZE
from ..constants import DocCategories as DocCat
from ..registry import register_handler
from ..structs import Chunk, DocAndContent, PageContent
from ..utils import calculate_image_tokens
from .base_chunker import BaseDocChunker

# Constants can be defined here or imported from a central config
MAX_CHUNK_IMG_NUM = 50
ENCODER = LLMUtils.get_encoding_for_model()


@register_handler
class PDFChunker(BaseDocChunker):
    """Handler for chunking PDF documents."""

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.PDF]

    async def chunk(self, doc_with_content: DocAndContent) -> list[Chunk]:
        """Splits a single PDF file into chunks based on pages."""
        if not isinstance(doc_with_content.content, bytes):
            return []

        doc_pages = await get_text_pages(doc_with_content.content)
        pages: list[PageContent] = []
        with open_pdf_document(doc_with_content.content) as pdf_document:
            for page_num, page_text in enumerate(doc_pages, 1):
                image_data = page_to_base64(pdf_document, page_num)
                text_tokens = len(ENCODER.encode(page_text.page_content))
                img_tokens = (
                    calculate_image_tokens(image_data) if image_data else 0
                )
                pages.append(
                    PageContent(
                        page_num,
                        [page_text.page_content],
                        [image_data] if image_data else [],
                        text_tokens + img_tokens,
                    )
                )

        merged_page_groups = aggregate_page_content(
            pages, ENCODER, MAX_CHUNK_SIZE, MAX_CHUNK_IMG_NUM
        )
        return self._create_chunks_from_page_groups(
            merged_page_groups, doc_with_content.doc
        )

    async def to_markdown(self, doc_with_content: DocAndContent) -> str:
        """Extracts all text from a PDF document into a single string."""
        if not isinstance(doc_with_content.content, bytes):
            return ""
        text_pages = await get_text_pages(doc_with_content.content)
        return "\n\n".join(p.page_content for p in text_pages if p.page_content)

    @staticmethod
    def _create_chunks_from_page_groups(
        groups: list[list[PageContent]], doc
    ) -> list[Chunk]:
        """Converts grouped page content into final Chunk objects."""
        doc_title = doc.name
        page_chunks = []
        for group in groups:
            texts = [
                f"Page {content.page_id}: {txt}"
                for content in group
                for txt in content.texts
            ]
            page_ids = sorted(list({content.page_id for content in group}))
            page_chunks.append(
                Chunk(
                    LCDoc("\n".join(texts)),
                    doc.id,
                    doc.url,
                    doc_title,
                    DocCat.PDF,
                    getattr(doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
                    group,
                    page_ids,
                )
            )
        return page_chunks


@register_handler
class WholePDFChunker(BaseDocChunker):
    """
    Handler for treating an entire PDF document as a single chunk.
    It extracts all pages and content but does not split them.

    Attributes:
        include_page_numbers (bool): If True, prepends "Page X: " to the
         text of each page. Defaults to False.
    """

    def __init__(self, include_page_numbers: bool = False):
        """
        Initializes the WholePDFChunker.

        Args:
            include_page_numbers (bool): Whether to include page
                numbers in the final chunk's text. Defaults to False.
        """
        super().__init__()
        self.include_page_numbers = include_page_numbers

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.PDF_AS_A_WHOLE]

    async def chunk(self, doc_with_content: DocAndContent) -> list[Chunk]:
        """
        Processes a single PDF file, treating the entire document as
        one chunk.
        """
        if not isinstance(doc_with_content.content, bytes):
            return []

        # 1. Extract all pages and their content
        doc_pages = await get_text_pages(doc_with_content.content)
        pages: list[PageContent] = []
        with open_pdf_document(doc_with_content.content) as pdf_document:
            for page_num, page_text in enumerate(doc_pages, 1):
                image_data = page_to_base64(pdf_document, page_num)
                text_tokens = len(ENCODER.encode(page_text.page_content))
                img_tokens = (
                    calculate_image_tokens(image_data) if image_data else 0
                )
                pages.append(
                    PageContent(
                        page_num,
                        [page_text.page_content],
                        [image_data] if image_data else [],
                        text_tokens + img_tokens,
                    )
                )

        # 2. Treat all pages as a single group
        merged_page_groups = [pages] if pages else []

        # 3. Create the single chunk from this group, passing the option
        return self._create_chunks_from_page_groups(
            merged_page_groups, doc_with_content.doc, self.include_page_numbers
        )

    async def to_markdown(self, doc_with_content: DocAndContent) -> str:
        """
        Extracts all text from a PDF document into a single string.
        """
        if not isinstance(doc_with_content.content, bytes):
            return ""
        text_pages = await get_text_pages(doc_with_content.content)
        return "\n\n".join(p.page_content for p in text_pages if p.page_content)

    @staticmethod
    def _create_chunks_from_page_groups(
        groups: list[list[PageContent]], doc, include_page_numbers: bool
    ) -> list[Chunk]:
        """Converts grouped page content into final Chunk objects."""
        doc_title = doc.name
        page_chunks = []
        for group in groups:
            if include_page_numbers:
                texts = [
                    f"Page {content.page_id}: {txt}"
                    for content in group
                    for txt in content.texts
                ]
            else:
                texts = [txt for content in group for txt in content.texts]

            page_ids = sorted(list({content.page_id for content in group}))
            page_chunks.append(
                Chunk(
                    LCDoc("\n".join(texts)),
                    doc.id,
                    doc.url,
                    doc_title,
                    DocCat.PDF_AS_A_WHOLE,
                    getattr(doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
                    group,
                    page_ids,
                )
            )
        return page_chunks

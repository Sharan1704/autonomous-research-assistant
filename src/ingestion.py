import glob
import logging
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Absolute default anchored to the project root so the path is valid
# regardless of the working directory at launch time.
_DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent / "data" / "sample_docs"
)


def _load_pymupdf_loader():
    """Lazy-import PyMuPDFLoader so a missing/broken pymupdf only breaks
    PDF loading, not the entire ingestion module."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required for PDF loading. "
            "Install it with: pip install pymupdf"
        ) from exc


class DataIngestion:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def load_single_text_file(self, file_path: str):
        """Load a single text file and return a list of Documents."""
        logger.debug("Loading text file: %s", file_path)
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.txt"):
        """Load all matching files in a directory."""
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(
                f"Directory not found: '{directory_path}'. "
                "Please check the path and try again."
            )
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()

    def load_pdf(self, file_path: str):
        """Load a single PDF file and return a list of Documents."""
        logger.debug("Loading PDF: %s", file_path)
        PyMuPDFLoader = _load_pymupdf_loader()
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def process_and_chunk(self, documents: list) -> list:
        """Split a list of Document objects into chunks."""
        return self.text_splitter.split_documents(documents)

    def ingest_data_folder(self, folder_path: str = _DEFAULT_DATA_DIR) -> list:
        """Ingest all supported files (.txt, .pdf) in a folder and chunk them."""
        documents = []

        # Load text files
        try:
            txt_docs = self.load_directory(folder_path, glob_pattern="**/*.txt")
            documents.extend(txt_docs)
            logger.info("Loaded %d text document(s) from '%s'.", len(txt_docs), folder_path)
        except FileNotFoundError as e:
            logger.error("%s", e)
            return []
        except Exception as e:
            logger.error("Error loading text files from '%s': %s", folder_path, e)

        # Load PDF files manually (DirectoryLoader can be unreliable with mixed types)
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        for pdf_file in pdf_files:
            try:
                pdf_docs = self.load_pdf(pdf_file)
                documents.extend(pdf_docs)
                logger.info("Loaded PDF: %s", pdf_file)
            except Exception as e:
                logger.error("Error loading PDF '%s': %s", pdf_file, e)

        if not documents:
            logger.warning("No documents found in '%s'.", folder_path)
            return []

        chunks = self.process_and_chunk(documents)
        logger.info(
            "Produced %d chunk(s) from %d document(s).", len(chunks), len(documents)
        )
        return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestion = DataIngestion()
    chunks = ingestion.ingest_data_folder()
    print(f"Generated {len(chunks)} chunks.")
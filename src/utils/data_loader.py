from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

# Load environment variables for any needed API keys
load_dotenv()

class MedicalDataLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Tim uses SentenceSplitter to keep medical sentences intact
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def load_and_chunk_pdf(self, file_path: str):
        """As seen in Screenshot 4: Extracts and chunks text from a PDF"""
        # Initialize the PDF reader
        reader = PDFReader()
        
        # Load the data from the PDF file
        docs = reader.load_data(file=file_path)
        
        # Screenshot 4 logic: Safely extract text from each document
        texts = [getattr(d, "text", "") for d in docs]
        
        chunks = []
        for t in texts:
            # Split the text into manageable chunks
            chunks.extend(self.splitter.split_text(t))
            
        return chunks
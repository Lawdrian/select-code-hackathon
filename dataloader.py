import webvtt
from langchain.schema import Document
from langchain_core.document_loaders.base import BaseLoader
import os
class VTTLoader(BaseLoader):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def load(self):
        documents = []
        # Get list of all VTT files in the folder
        vtt_files = [f for f in os.listdir(self.file_path) if f.endswith('.vtt')]

        for file_name in vtt_files:
            combined_content = ""
            file_path = os.path.join(self.file_path, file_name)
            for caption in webvtt.read(file_path):
                combined_content += caption.text + "\n"
            documents.append(Document(page_content=combined_content))
        return documents

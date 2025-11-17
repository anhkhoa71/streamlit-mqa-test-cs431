import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFPreprocessor:
    def __init__(
        self,
        chunk_size=400,
        chunk_overlap=100,
        separators=None,
        min_tokens=20,
        min_alpha_ratio=0.7,
        min_avg_words_per_sentence=3
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? "]
        self.min_tokens = min_tokens
        self.min_alpha_ratio = min_alpha_ratio
        self.min_avg_words_per_sentence = min_avg_words_per_sentence

    def clean_text(self, text):
        text = re.sub(r"\n\s*\n", "\n", text)
        text = re.sub(r"[^\w\s.,;:?!\u00C0-\u017F]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_noise(self, text):
        t = text.lower()
        if len(text.strip()) < 30:
            return True
        if "mục lục" in t:
            return True
        if re.fullmatch(r"[\d\.\s]+", text):
            return True
        return False

    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        for d in docs:
            d.page_content = self.clean_text(d.page_content)
        
        return docs

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        return splitter.split_documents(documents)

    def filter_by_heuristics(self, chunks):
        filtered = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, c in enumerate(chunks):
            text = c["text"]
            total_chars = len(text)
            if total_chars == 0:
                continue
            
            alpha_chars = len(re.findall(r"[A-Za-zÀ-ỹ]", text))
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < self.min_alpha_ratio:
                continue
            
            sentences = re.split(r"[.!?]\s+", text)
            avg_words = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if avg_words < self.min_avg_words_per_sentence:
                continue
            
            filtered.append(c)
            progress_bar.progress((idx + 1) / len(chunks))
            status_text.text(f"Đang lọc chunks: {idx + 1}/{len(chunks)}")
        
        progress_bar.empty()
        status_text.empty()
        
        for i, c in enumerate(filtered):
            c["chunk_id"] = i
            c["total_chunks"] = len(filtered)
        
        return filtered

    def process(self, pdf_path):
        docs = self.load_pdf(pdf_path)
        raw_chunks = self.split_documents(docs)

        chunks = []
        for i, c in enumerate(raw_chunks):
            txt = c.page_content
            if not self.is_noise(txt) and len(txt.split()) >= self.min_tokens:
                chunks.append({
                    "chunk_id": i,
                    "text": txt,
                    "source_page": c.metadata.get("page"),
                    "total_chunks": len(raw_chunks)
                })

        chunks = self.filter_by_heuristics(chunks)
        return chunks
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF


class MCQGenerator:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate_question(self, input_text, max_length=512):
        prompt = (
            "mcq:\n"
            "Nhiệm vụ: Sinh một câu hỏi trắc nghiệm 4 lựa chọn (A, B, C, D) và dáp án kèm theo"
            " phù hợp với nội dung đoạn văn dưới đây.\n"
            "Lưu ý: Câu hỏi có thể là bất kỳ câu hỏi nào miễn là hợp lý.\n\n"
            "Đoạn văn:\n"
        )
        input_text = prompt + input_text.strip()
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.6,
                no_repeat_ngram_size=3
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = re.sub(r"\s*(A\.|B\.|C\.|D\.)", r"\n\1", decoded).strip()
        decoded = re.sub(r"\s*(Đáp án:)", r"\n\1", decoded)
        return decoded

    def generate_from_chunks(self, chunks):
        outputs = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, chunk in enumerate(chunks):
            output = self.generate_question(chunk['text'])
            output += f"\nVị trí: trang {chunk['source_page']}"
            outputs.append(output)
            
            progress_bar.progress((idx + 1) / len(chunks))
            status_text.text(f"Đang sinh câu hỏi: {idx + 1}/{len(chunks)}")
        
        progress_bar.empty()
        status_text.empty()
        return outputs

    def export_to_pdf(self, outputs, pdf_path="quiz_output.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(0, 10, "ĐỀ KIỂM TRA TRẮC NGHIỆM\n\n")
        
        for i, out in enumerate(outputs, 1):
            pdf.multi_cell(0, 10, f"Câu {i}:")
            pdf.multi_cell(0, 10, out + "\n")
        
        pdf.output(pdf_path)
        return pdf_path
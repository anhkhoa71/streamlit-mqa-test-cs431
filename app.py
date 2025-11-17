import streamlit as st
import os
import tempfile
import torch
import gc
from preprocess import PDFPreprocessor
from generator import MCQGenerator


def main(model_path="anhkhoa71/viT5-ViQuaD"):
    st.set_page_config(
        page_title="MCQ Generator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🎓 Hệ thống Sinh Câu hỏi Trắc nghiệm Tự động")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        st.subheader("📊 Tham số Preprocessing")
        chunk_size = st.slider("Chunk Size", 200, 800, 400, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 25)
        min_tokens = st.slider("Min Tokens", 10, 50, 20, 5)
        
        st.subheader("🔧 Device")
        device = st.radio(
            "Chọn device:",
            ["cuda", "cpu"],
            index=0 if torch.cuda.is_available() else 1
        )
        
        if device == "cuda" and not torch.cuda.is_available():
            st.warning("⚠️ CUDA không khả dụng, sẽ sử dụng CPU")
            device = "cpu"
        
        st.info(f"💻 Đang sử dụng: **{device.upper()}**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload File PDF")
        uploaded_file = st.file_uploader(
            "Chọn file PDF",
            type=['pdf'],
            help="Upload file PDF để sinh câu hỏi trắc nghiệm"
        )
        
        if uploaded_file:
            st.success(f"✅ Đã upload: {uploaded_file.name}")
            st.info(f"📦 Kích thước: {uploaded_file.size / 1024:.2f} KB")
    
    with col2:
        st.subheader("📋 Thông tin xử lý")
        if uploaded_file:
            st.metric("Tên file", uploaded_file.name)
            st.metric("Kích thước", f"{uploaded_file.size / 1024:.2f} KB")
    
    st.markdown("---")
    
    if uploaded_file:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            process_button = st.button(
                "🚀 Bắt đầu Xử lý",
                type="primary",
                use_container_width=True
            )
        
        if process_button:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_pdf_path = tmp_file.name
                
                st.subheader("🔄 Bước 1: Tiền xử lý PDF")
                with st.spinner("Đang xử lý PDF..."):
                    preprocessor = PDFPreprocessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        min_tokens=min_tokens
                    )
                    chunks = preprocessor.process(tmp_pdf_path)
                
                st.success(f"✅ Hoàn thành! Tổng số chunks: {len(chunks)}")
                
                with st.expander("👀 Xem mẫu chunks"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(f"**Chunk {i+1}** (Trang {chunk['source_page']}):")
                        st.text(chunk['text'][:200] + "...")
                        st.markdown("---")
                
                st.subheader("🤖 Bước 2: Sinh câu hỏi trắc nghiệm")
                with st.spinner("Đang load model..."):
                    generator = MCQGenerator(model_path, device=device)
                
                st.info("⚡ Đang sinh câu hỏi...")
                outputs = generator.generate_from_chunks(chunks)
                
                st.success(f"✅ Đã sinh {len(outputs)} câu hỏi!")
                
                st.subheader("📝 Kết quả")
                with st.expander("🔍 Xem tất cả câu hỏi", expanded=True):
                    for i, output in enumerate(outputs, 1):
                        st.markdown(f"### Câu {i}")
                        st.text(output)
                        st.markdown("---")
                
                st.subheader("💾 Bước 3: Xuất file PDF")
                with st.spinner("Đang tạo file PDF..."):
                    output_pdf_path = tempfile.mktemp(suffix='.pdf')
                    generator.export_to_pdf(outputs, output_pdf_path)
                
                with open(output_pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                    
                    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                    with col_dl2:
                        st.download_button(
                            label="⬇️ Tải xuống Đề thi",
                            data=pdf_bytes,
                            file_name=f"de_thi_{uploaded_file.name.replace('.pdf', '')}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                
                st.success("🎉 Hoàn thành toàn bộ quy trình!")
                
                os.unlink(tmp_pdf_path)
                os.unlink(output_pdf_path)
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Vui lòng upload file PDF để bắt đầu")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>📚 Hệ thống Sinh Câu hỏi Trắc nghiệm Tự động</p>
            <p>Powered by ViT5 & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
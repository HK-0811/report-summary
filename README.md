# Medical Report Diagnosis Streamlit App 👨‍⚕️

### Overview
This Streamlit application is designed to assist in the analysis and diagnosis of medical reports, specifically pathology laboratory blood test reports. The app allows users to upload medical reports in either JPG/JPEG or PDF formats. Once uploaded, the application extracts text from the report, summarizes it using a powerful language model, and provides an interface for users to ask specific questions about the report's content.

## Features
- File Upload: Supports uploading medical reports in JPG, JPEG, and PDF formats.
- Text Extraction:
  - For JPG/JPEG files: Optical Character Recognition (OCR) is performed using easyocr to extract text from the images.
  - For PDF files: Text is directly extracted using PyMuPDF.
- Text Summarization: Utilizes the meta-llama/Meta-Llama-3-8B-Instruct language model from Hugging Face to generate concise summaries of the extracted text.
- Question-Answer Interface: Allows users to interact with the summarized report by asking specific questions. The app uses a FAISS vector store to retrieve relevant information from the text and provide accurate answers.

## How It Works

1. Upload Report: Users can upload a medical report in JPG, JPEG, or PDF format.
2. Text Extraction:
   - Images are processed with easyocr to extract textual content.
   - PDFs are processed with PyMuPDF to extract text directly.
3. Text Summarization:
   - The extracted text is split into manageable chunks using langchain's RecursiveCharacterTextSplitter.
   - These chunks are then summarized using the meta-llama/Meta-Llama-3-8B-Instruct language model.
4. Interactive Q&A:
   - The text chunks are embedded into a vector store using FAISS and HuggingFaceEmbeddings.
   - Users can type questions about the report, and the app retrieves the most relevant information to provide a concise answer.



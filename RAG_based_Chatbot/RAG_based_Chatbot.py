#%pip install langchain-community
#%pip install langchain-text-splitter
#%pip install langchain-groq
#%pip install langchain-huggingface
#%pip install langchain-classic
#%pip install gradio


import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load variables from .env
load_dotenv()


# Global QA chain
qa_chain = None


# ==================================================
# 1. Upload PDF and create RAG
# ==================================================

def process_pdf(pdf_file):
    global qa_chain

    if pdf_file is None:
        return "Please upload a PDF file first."

    api_key = os.getenv("GROQ_API_KEY","")

    if not api_key:
        return (
            "GROQ_API_KEY was not found. "
            "Add it to your .env file and restart the app."
        )

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "No readable pages were found in this PDF."

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            return "No text could be extracted from this PDF."

        # Free local embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            #model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Create FAISS vector database
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Retrieve the 5 most relevant chunks
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # Free-tier Groq chat model
        llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
            api_key=api_key
        )

        # Prompt to answer using the PDF context
        prompt_template = """
        Answer the question using ONLY the context below.
        If the answer is not present in the context, say:
        "I could not find this information in the PDF."

        Context:
        {context}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create RAG chain
        new_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Update only after successful processing
        qa_chain = new_qa_chain

        return (
            "### PDF processed successfully!\n\n"
            f"- **Pages:** {len(documents)}\n"
            f"- **Chunks:** {len(chunks)}\n\n"
            "You can now ask questions about your PDF."
        )

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


# ==================================================
# 2. Ask questions
# ==================================================

def ask_pdf(question):
    global qa_chain

    if qa_chain is None:
        return "Please upload and process a PDF first."

    if not question or not question.strip():
        return "Please enter a question."

    try:
        response = qa_chain.invoke({"query": question})

        answer = response["result"]
        source_documents = response.get("source_documents", [])

        # Collect unique page numbers (PyPDFLoader starts at 0)
        pages = sorted({
            doc.metadata["page"] + 1
            for doc in source_documents
            if doc.metadata.get("page") is not None
        })

        if pages:
            answer += "\n\n---\n"
            answer += "**Source pages:** " + ", ".join(map(str, pages))

        return answer

    except Exception as e:
        return f"Error answering question: {str(e)}"


# ==================================================
# 3. Gradio interface
# ==================================================

with gr.Blocks(title="Free PDF RAG Assistant") as demo:

    gr.Markdown(
        """
        # PDF RAG Assistant

        Upload a PDF, process it, and ask questions.

        **LLM:** Groq | **Embeddings:** Hugging Face (local)
        """
    )

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button("Process PDF")
    status = gr.Markdown()

    process_button.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )

    gr.Markdown("---")

    question = gr.Textbox(
        label="Ask a question",
        placeholder="Example: Summarize this PDF"
    )

    ask_button = gr.Button("Ask PDF")
    answer = gr.Markdown(label="Answer")

    ask_button.click(
        fn=ask_pdf,
        inputs=question,
        outputs=answer
    )

    question.submit(
        fn=ask_pdf,
        inputs=question,
        outputs=answer
    )


# ==================================================
# 4. Launch app
# ==================================================

if __name__ == "__main__":
    demo.launch()

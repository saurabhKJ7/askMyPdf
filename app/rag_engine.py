from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    model_name="deepseek-r1-distill-qwen-7b",
    temperature=0.7,
    openai_api_key="sk-anything"
)

vectorstore = None
qa_chain = None

def load_pdf_to_vectorstore(file_path: str):
    global vectorstore, qa_chain

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # ✅ Use the recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str) -> str:
    if not qa_chain:
        return "No document uploaded yet."
    return qa_chain.run(query)

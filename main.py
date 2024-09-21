# Import the necessary libraries
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF document
loader = PyPDFLoader("/Users/dad/Downloads/ai_adoption_framework_whitepaper.pdf")

# Extract the document
docs = loader.load()

# Optionally filter out unwanted content (e.g., very short chunks)
filtered_docs = [doc for doc in docs if len(doc.page_content) > 100]

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)

# Split the filtered documents into chunks
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings )

llm = OllamaLLM(model = "llama3.1")

chain = RetrievalQA.from_chain_type(
    llm,
    retriever = db.as_retriever()
)

question = "what is the document is all about?"

results = chain.invoke({"query":question})

print(results)

# # Print chunks to inspect their content
# for i, text in enumerate(texts):
#     print(f"Text chunk {i}: {text}")


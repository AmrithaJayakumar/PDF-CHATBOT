from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

query="YOLOv7 outperforms which models"

HF_REPO_NAME = "TheBloke/Llama-2-13B-chat-GGUF"
HF_MODEL_NAME = "llama-2-13b-chat.Q4_K_S.gguf"
model_path = hf_hub_download(repo_id=HF_REPO_NAME, filename=HF_MODEL_NAME)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '90661a70-799a-4edf-8eac-f9528783ab46')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

import os
os.environ["PINECONE_API_KEY"] ="90661a70-799a-4edf-8eac-f9528783ab46"


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

loader = PyPDFLoader("yolo.pdf")
data = loader.load()
docs=text_splitter.split_documents(data)

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



from langchain.vectorstores import Pinecone as PC
docs_chunks = [t.page_content for t in docs]
docsearch = PC.from_texts(
    docs_chunks,
    embeddings,
    index_name='chatbot'
)

n_gpu_layers = 40
n_batch = 256
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=True,
)

chain=load_qa_chain(llm, chain_type="stuff")
docs=docsearch.similarity_search(query)

c=chain.run(input_documents=docs, question=query)
print(docs)
print(c)

print("completed execution")
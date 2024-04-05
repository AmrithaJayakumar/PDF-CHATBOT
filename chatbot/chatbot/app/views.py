from django.shortcuts import render,redirect
from .models import *
from django.contrib.auth.models import User,auth
from django.core.files.storage import FileSystemStorage
from django.conf import settings


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
from langchain.vectorstores import Pinecone as PC

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



embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



# from langchain.vectorstores import Pinecone as PC
# docs_chunks = [t.page_content for t in docs]
# docsearch = PC.from_texts(
#     docs_chunks,
#     embeddings,
#     index_name='chatbot'
# )

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



# Create your views here.
def reg(request):
    if request.method=='POST':
        name=request.POST['uname']
        email=request.POST['mail']
        password=request.POST['p_password']
        confirm_password=request.POST['pp_password']
        if password==confirm_password:
            if User.objects.filter(username=name).exists():
                context = {
                'msg':'user already exists....'
                }
                return render(request,'register.html',context)
            else:
                User.objects.create_user(username=name,email=email,password=password).save()
               
                # context = {
                # 'msg':'successfully registered.'
                # }
                return redirect(reg)
        else:
            context = {
                'msg':'password doesnot match....'
            }
            return render(request,'regiter.html',context)
    return render (request,'regiter.html')

def login(request):
    if request.method=='POST':
        UserName=request.POST['uname']
        Password=request.POST['p_password']   
        if User.objects.filter(username=UserName).exists():
            u=auth.authenticate(username=UserName,password=Password)
            if u is not None:
                auth.login(request,u)
                return redirect(home)
            else:
                context = {
                    'key':'invalid user'
                }
                return render(request,'login.html',context)
        else:
            context = {
                'key':'invalid user'
            }
            return render(request,'login.html',context)    
    return render(request,'login.html')

def home(request):
    a=User.objects.get(username=request.user)
    context={
        'key':a
    }
    if request.method == 'POST' and request.FILES.get('fileUpload'):
        uploaded_file = request.FILES['fileUpload']
        question = request.POST['question']
        print(question)
        fs = FileSystemStorage(location='app\static\media')
        print(fs)
        filename = fs.save(uploaded_file.name,uploaded_file)
        print(filename)
        
        media_path = settings.MEDIA_ROOT
        f=os.path.join(media_path, filename)
        
        loader = PyPDFLoader(f)
        data = loader.load()
        docs=text_splitter.split_documents(data)
        
        docs_chunks = [t.page_content for t in docs]
        docsearch = PC.from_texts(
            docs_chunks,
            embeddings,
            index_name='chatbot'

        docs=docsearch.similarity_search(query)
        c=chain.run(input_documents=docs, question=query)
        print(c)
        return redirect(home)
    return render (request,'home.html',context)
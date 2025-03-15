from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

# Delete existing Chroma database folder
shutil.rmtree("chroma_db", ignore_errors=True)

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm = ChatGroq(api_key = api_key, model_name = 'gemma2-9b-it')



embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
def processing(pdf):
    loader = TextLoader(pdf)
    #loader = DirectoryLoader('law_data', glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    chuncks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    db = Chroma.from_documents(chuncks, embedding, persist_directory="./chroma_db")
    return db

template = """you are an egyption lawyer with 20 years experience in the feild. 
You are currently working as a legal consultant for a law firm in Cairo. 
please answer all related question based on the provided context:  {context}
question:{input}"""

prompt = PromptTemplate(
    input_variables=['context', 'input'],
    template = template
) 

db = processing('text.txt')

retriever = db.as_retriever()
doc_chain = create_stuff_documents_chain(llm, prompt)
ret_chain = create_retrieval_chain(retriever, doc_chain)



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# Define request model
class QuestionRequest(BaseModel):
    question: str


@app.post("/response")
async def get_response(request: QuestionRequest):
    response = ret_chain.invoke({'input': request.question})
    return {"answer": response['answer']}



#answer = ret_chain.invoke({'input': 'what are the rights of education'})
#print(answer['answer'])
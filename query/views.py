from django.shortcuts import render, redirect
from django.http import HttpResponse
import markdown2

# Create your views here.

from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
import uuid
import base64
import fitz
from PIL import Image
import io
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

load_dotenv() # Load environment variables from .env file

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model_image = genai.GenerativeModel("gemini-1.5-flash")
model_documents = GoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25")
embeddings_model =  GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# request.session["chat_history"] = []

def load_vectors(request):
    loader = PyPDFLoader(
        "Gemini/Babok Guide.pdf",
        mode="page",
        images_inner_format="markdown-img",
        images_parser=LLMImageBlobParser(model=GoogleGenerativeAI(model="gemini-1.5-flash")),
    )

    # file = PyPDFLoader("Gemini/Babok Guide.pdf")
    docs = loader.load()
    # print(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_list = text_splitter.split_documents(docs)

   
    text_vectors = Chroma.from_documents(text_list, embeddings_model, persist_directory= "./query/chroma_db")
    
def query(request):
    if request.method == 'POST':
        # load_vectors(request)
        query_request = request.POST.get('query')
        return redirect('query_result', query_request=query_request)
    return render(request, 'query/query.html', {'message': 'Please submit your query.'})
    
def query_result(request,query_request):
    request.session.modified = True
    session_items = request.session.get('answer', [])
    session_question = request.session.get('question', [])
    query = query_request
    db = Chroma(persist_directory="./query/chroma_db", embedding_function=embeddings_model)
    prompt = ChatPromptTemplate.from_template("""
                                        Please search the answer on basis of context provided.
                                        
                                        {context}
                                        Answer the question based on the context provided.
                                        Question: {input}
                                        
                                        If answer is not found in the context, please say "I don't know. 
                                        The BABook does not cover this topic".
                                        """)

    document_chain = create_stuff_documents_chain(llm = model_documents, prompt=prompt)


    db_retreiver = db.as_retriever()
    retreiver_chain = create_retrieval_chain(retriever=db_retreiver, combine_docs_chain=document_chain)

    response = retreiver_chain.invoke({"input": query})
    formatted_html = markdown2.markdown(response["answer"])
    
    session_items.append(formatted_html)
    session_question.append(query_request)
    
    return render(request, 'query/index.html', {'response': formatted_html,
                                                "question": query_request
                                                , "session_items": session_items,
                                                "session_question": session_question})

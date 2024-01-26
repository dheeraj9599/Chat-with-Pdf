# Necessary libraries

import streamlit as st
# for using Stickers and animations
from streamlit_lottie import st_lottie

import requests

# Necessary LangChain components

# from langchain.vectorstores.cassandra import Cassandra # Database to store Vectors
from langchain.indexes.vectorstore import VectorStoreIndexWrapper # For wrapping the vectors in a package
from langchain.vectorstores import FAISS # For Storing the Vectors 
from langchain.llms import OpenAI # LLM Model
from langchain.embeddings import OpenAIEmbeddings # For converting text into vectors 
from PyPDF2 import PdfReader # For reading a PDF Document
# from typing_extensions import Concatenate # For Concatenating Strings
from langchain.text_splitter import CharacterTextSplitter # for splitting words into characters


OPENAI_API_KEY = st.secrets["API_KEY"]


llm = OpenAI(openai_api_key = OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)


def Model():
  # The below-mentioned code has already been executed once.

  #   # reading PDF
  #   pdfreader = PdfReader('48lawsofpower.pdf')
  
  #   raw_text='' # read text from pdf
  #   for i, page in enumerate (pdfreader.pages):
  #     content = page.extract_text()
  #     if content:
  #       raw_text += content 

  
  
  # splitting the text using Character Text Split to minimize token size
  #  text_splitter = CharacterTextSplitter(
  #    separator = "\n",
  #    chunk_size = 1000,
  #    chunk_overlap = 200,
  #    length_function = len,
  #  )  
   
  #  texts = text_splitter.split_text(raw_text)

  #  astra_vector_store.add_texts(texts)
   
  #  vector_store = FAISS.from_texts(texts, embedding=embedding)
   
  #  FAISS.save_local("vector") # saving the vector
   
   vector_store = FAISS.load_local("vector", embeddings=embedding)

   vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
  
   return vector_index


def QNA(question):

  if question == "":
    return ""
  
  vector_index = Model()

  query_text = question
  answer = vector_index.query(query_text, llm=llm).strip()
  return answer


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    
    # page metadata
    st.set_page_config(page_title="Chat Bot", page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/400px-PDF_file_icon.svg.png", layout="wide")
    
    welcome = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_1TcivY.json")   
    st_lottie(welcome, height=150, key="Welcome" )
    
    st.header("Hello I am a AI Bot :wave:") 

    book = "48 laws of power"
    st.subheader("I will answer your questions from the book '48 laws of power'")   

    question = st.text_input("Ask your Question?")

    if st.button('Get Answer'):
      response = QNA(question)
      st.balloons()

      if response == "":
        st.write("Please Enter the Question")
      else:
        st.write(response)

if __name__ == '__main__':
         main()
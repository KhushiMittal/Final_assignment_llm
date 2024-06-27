import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotted_dict import DottedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
load_dotenv()

azure_config = {
    "base_url": os.getenv("DONO_AZURE_OPENAI_BASE_URL"),
    "model_deployment": os.getenv("DONO_AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "model_name": os.getenv("DONO_AZURE_OPENAI_MODEL_NAME"),
    "embedding_deployment": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    "embedding_name": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_NAME"),
    "api_key": os.getenv("DONO_AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("DONO_AZURE_OPENAI_API_VERSION")
    }

st.title("WIKIPEDIA RAG")
@st.cache_resource(show_spinner=False)
def create_models(azure_config):
    models=DottedDict()
    llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api_key"],
                      openai_api_version=azure_config["api_version"],
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_config["api_key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["base_url"],
        model = azure_config["embedding_deployment"]
    )
    models.llm=llm
    models.embedding_model=embedding_model 
    return models

@st.cache_resource(show_spinner=False)
def load_from_wikipedia(query):
    loader = WebBaseLoader(web_path=query)
    data = loader.load()
    return data

def chunk_data(data):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter =RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, is_separator_regex=False, separators=["."])
    chunks = text_splitter.split_documents(data)
    return chunks

def get_vector_store(text_chunks, models):
    embeddings = models.embedding_model
    vectordb=FAISS.from_documents(chunked,embeddings)
    return vectordb

def get_conversational_chain(llm):
    prompt_template = """
    You are a helpful bot. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def get_answer(query):
    document_search=get_vector_store(chunked, models)
    similar_docs = document_search.similarity_search(query, k=1) # get closest chunks
    chain = get_conversational_chain(models.llm)
    answer = chain.invoke(input={"input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer

models=create_models(azure_config)
doc=load_from_wikipedia('https://en.wikipedia.org/wiki/FIFA_World_Cup')
chunked=chunk_data(doc)

with st.form("my_form"):
    query = st.text_area("Ask Question from the wikipedia link: ")
    submitted = st.form_submit_button("Submit Query")
    if submitted:
        response = get_answer(query)
        st.write(response['output_text'])





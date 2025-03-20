import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from docx import Document as DC
import time
import secrets


# Streamlit UI
st.set_page_config(page_title="SEO Content Generator", page_icon="🤓", layout="wide")

st.title("SEO Content Generator")

# Add a button to connect on LinkedIn at the right side of the header
col1, col2 = st.columns([6, 1])
with col2:
    st.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/abhishek-kumawat-iitd/)", unsafe_allow_html=True)

st.sidebar.header("Input Parameters")
doc_file = "seo_content_revamp_AK.docx"

generated_content = None
regenerated_content = None

# User Inputs
language = st.sidebar.text_input("Language", "Danish")
country = st.sidebar.text_input("Country", "Denmark")
brand = st.sidebar.text_input("Brand", "Danland")
base_urls = st.sidebar.text_area("Base URLs (comma-separated)").split(",")
additional_input = st.sidebar.text_area("Additional Input (Enter additional input here)")

# Initialize AI Clients
ai_client = AzureChatOpenAI(
    api_key=st.secrets["openai_api"],
    api_version="2023-12-01-preview",
    azure_endpoint="https://akash-seo.openai.azure.com/",
    azure_deployment="gpt-data"
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
encoder = SentenceTransformer("all-mpnet-base-v2")

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return " ".join([p.get_text() for p in soup.find_all("p")])
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching {url}: {e}")
        return ""

def generate_keyword(text):
    query = f"Based on the {text}. Tell me a keyword which sums up all content."
    response = ai_client.invoke(query)
    return response.content.strip()

def scrape_google_search_results(keyword, num_results=5):
    search = GoogleSearch({"q": keyword, "num": num_results, "api_key": st.secrets["serpapi_api"]})
    results = search.get_dict()
    return [result.get("link") for result in results.get("organic_results", [])[:num_results]]

def extract_texts_from_urls(url_list):
    return " ".join([extract_text_from_url(url) for url in url_list if url])

def extract_clean_text(documents):
    return " ".join([doc.page_content for doc in documents])

def generate_seo_content(text, keyword):
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    data = [Document(page_content=text)]
    docs = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vec = encoder.encode(keyword)
    retriever = vectorstore.as_retriever(score_threshold=0.7)
    rdocs = retriever.invoke(keyword)
    cleaned_text = extract_clean_text(rdocs)
    query = f"Generate optimized SEO content for {brand} in {language} considering the country : ({country}) which can rank us on No.1. Include: Title (60w), Meta (160w), H1 (30w), Intro (40w), Info (600-800w). Avoid competitors."
    guideline = f"The blog should be in conversational tone for {brand}'s website."
    guideline2 = f"You should pick up the activities, events and generic facts about location from the {cleaned_text}. Write it as a human would write "
    response = ai_client.invoke(query + guideline + guideline2 + additional_input)
    return response.content.strip()

if st.sidebar.button("Generate SEO Content"):
    with st.spinner("Generating SEO Content..."):
        doc = DC()
        for idx, base_url in enumerate(base_urls):
            base_text = extract_text_from_url(base_url)
            keyword = generate_keyword(base_text)
            top_urls = scrape_google_search_results(keyword)
            competitor_text = extract_texts_from_urls(top_urls)
            generated_content = generate_seo_content(competitor_text, keyword)
            st.subheader(f"SEO Content for {base_url}")
            st.text(generated_content)
            doc.add_paragraph(f"URL: {base_url}\n{generated_content}")
        doc.save(doc_file)
        st.success("SEO Content document saved!")

        with open(doc_file, "rb") as file:
            st.download_button(
                label="Download Generated Document",
                data=file,
                file_name="seo_content_revamp.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

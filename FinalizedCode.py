import os
import json
import numpy as np
import faiss
import streamlit as st

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_vertexai import ChatVertexAI
from tenacity import retry, stop_after_attempt, wait_exponential


# Load environment variables
load_dotenv()

# Set environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GoogleVertex_API_KEY = os.getenv("GOOGLE_VERTEX_API_KEY")
if not COHERE_API_KEY or not GoogleVertex_API_KEY:
    raise ValueError("API Key(s) are not set in environment variables.")


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mymlproject-444721-140c4261cfb8.json"

# Token estimation function
def estimate_tokens(text):
    words = text.split()
    return int(len(words) / 0.75)  # Rough token estimate per document


# Process Uploaded Files
def process_uploaded_files(uploaded_files):
    """Process and load documents from uploaded files."""
    document_list = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        with NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Choose loader based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")  # Use UTF-8 encoding for text files
        else:
            st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}")
            continue

        # Load documents and extend the document list
        try:
            documents = loader.load()
            document_list.extend(documents)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue

        # Optionally clean up the temporary file after processing
        os.remove(temp_file_path)

    return document_list


# Split Documents into Chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20, max_tokens=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # Ensure chunks do not exceed token limit
    valid_chunks = []
    for chunk in chunks:
        token_count = estimate_tokens(chunk.page_content)
        if token_count <= max_tokens:
            valid_chunks.append(chunk)
        else:
            st.warning(f"Skipping chunk due to token limit: {chunk.page_content[:100]}... (tokens: {token_count})")

    return valid_chunks


# Create the Cohere Embedding Model
#embeddings = CohereEmbeddings(api_key=cohere_api_key, model="embed-english-v3.0")
try:
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
except Exception as e:
    st.error(f"Error initializing CohereEmbeddings: {e}")
    raise

# Implement Retry Logic for Embeddings
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
def embed_documents_with_retry(texts, embeddings):
    return embeddings.embed_documents(texts)


# Create FAISS Vectorstore from Docs
def create_faiss_vectorstore_from_docs(docs, embeddings, faiss_index_path, metadata_path):
    texts = [doc.page_content for doc in docs]

    # Batch embedding to prevent rate limit issues
    embeddings_matrix = []
    batch_size = 100  # Adjust based on rate limit
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings_matrix.extend(embed_documents_with_retry(batch, embeddings))

    embeddings_array = np.array(embeddings_matrix).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Overwrite FAISS index if it exists
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
    faiss.write_index(index, faiss_index_path)

    # Save updated metadata
    metadata = [{'doc_id': i, 'content': doc.page_content} for i, doc in enumerate(docs)]
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return index


# Perform Retrieval and Generate Answer
def search_similar_docs_with_faiss_and_generate_answer(
    query, index, metadata, embeddings, model, k=3, max_context_tokens=2000
):
    """Retrieve and synthesize information from multiple documents, prioritizing by relevance score."""
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k)

    if indices[0].size > 0 and np.any(indices[0] != -1):  # Ensure valid retrieval
        # Pair retrieved document indices with their relevance scores (distances)
        retrieved_docs = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:
                retrieved_docs.append((metadata[idx], score))

        # Deduplicate the retrieved documents
        unique_docs = {doc['content']: (doc, score) for doc, score in retrieved_docs}.values()
        retrieved_docs = list(unique_docs)

        # Sort documents by relevance (ascending distance means higher relevance)
        retrieved_docs.sort(key=lambda x: x[1])

        st.write(f"Retrieved {len(retrieved_docs)} document(s) for the query, sorted by relevance:")
        #for i, (doc, score) in enumerate(retrieved_docs):  # Display snippets and scores
        #    st.write(f"Document {i + 1} | Score: {score:.4f} | Snippet: {doc['content'][:200]}...")

        # Concatenate the most relevant context for the LLM
        context = "\n---\n".join(
            [f"Document {i + 1} (Score: {score:.4f}):\n{doc['content']}" for i, (doc, score) in enumerate(retrieved_docs)]
        )

        # Summarize context if it exceeds token limit
        if len(context.split()) > max_context_tokens:
            st.write("Context too large; summarizing the top documents...")
            context = summarize_long_context([doc for doc, _ in retrieved_docs], model, max_context_tokens)

        if not context.strip():
            st.write("No sufficient context retrieved to answer the query.")
            return "I cannot determine this from the provided information."

        # Generate the answer
        # st.write(f"Context passed to LLM: {context[:500]}...")  # Debugging output
        answer = generate_answer_with_llm(query, [doc for doc, _ in retrieved_docs], model)
        st.write(f"### Generated Answer: {answer}")
        return answer
    else:
        st.write("No relevant documents found for the query.")
        return "I cannot determine this from the provided information."




# Generate Answer Using LLM
def generate_answer_with_llm(query, retrieved_docs, model):
    context = "\n---\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are a helpful and advanced assistant designed to process information from provided documents and you can reason across multiple documents.
                    Your task is to answer the question based on the provided documents. If the answer cannot be determined from the documents, respond with "I cannot determine this from the provided information."

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )

    llm_chain = LLMChain(prompt=prompt, llm=model)
    answer = llm_chain.run({"question": query, "context": context})
    return answer


# Summarize Long Context to Fit Token Limit
def summarize_long_context(retrieved_docs, model, max_context_tokens=2000):
    """Summarize documents to fit within the token limit."""
    summaries = []
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Summarize the following document to capture the key points in 200 words or less:

        {content}

        Summary:"""
    )

    llm_chain = LLMChain(prompt=prompt, llm=model)

    for doc in retrieved_docs:
        summary = llm_chain.run({"content": doc['content']})
        summaries.append(summary.strip())

    # Combine summaries, ensuring they fit within the token limit
    combined_summary = "\n---\n".join(summaries)
    return combined_summary[:max_context_tokens]

# Process Uploaded Files and Update FAISS Index
def update_index_with_new_files(uploaded_files, faiss_index_path, metadata_path, embeddings):
    if uploaded_files:
        # Process newly uploaded documents
        new_docs = process_uploaded_files(uploaded_files)
        new_docs = split_docs(new_docs, chunk_size=1000, chunk_overlap=20, max_tokens=2000)

        if new_docs:
            # Load existing index and metadata if they exist
            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                st.write("Updating existing FAISS index...")
                existing_index = faiss.read_index(faiss_index_path)
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                st.write("Creating a new FAISS index...")
                existing_index = None
                existing_metadata = []

            # Embed new documents
            new_texts = [doc.page_content for doc in new_docs]
            new_embeddings = []
            for i in range(0, len(new_texts), 100):  # Batch embedding
                batch = new_texts[i:i + 100]
                new_embeddings.extend(embed_documents_with_retry(batch, embeddings))

            new_embeddings_array = np.array(new_embeddings).astype('float32')

            # Create or update FAISS index
            if existing_index:
                existing_index.add(new_embeddings_array)
                metadata = existing_metadata + [{'doc_id': len(existing_metadata) + i, 'content': doc.page_content}
                                                 for i, doc in enumerate(new_docs)]
            else:
                dimension = new_embeddings_array.shape[1]
                existing_index = faiss.IndexFlatL2(dimension)
                existing_index.add(new_embeddings_array)
                metadata = [{'doc_id': i, 'content': doc.page_content} for i, doc in enumerate(new_docs)]

            # Save the updated FAISS index and metadata
            faiss.write_index(existing_index, faiss_index_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            return existing_index, metadata
    return None, None


# Load FAISS Index and Metadata
def load_faiss_index(faiss_index_path):
    try:
        index = faiss.read_index(faiss_index_path)
        return index
    except Exception as e:
        st.write(f"Error loading FAISS index: {e}")
        return None

def load_metadata(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.write(f"Error loading metadata: {e}")
        return []


# Main Streamlit Interface
def main():
    st.title("Document-based Question Answering System")

    # Upload Documents
    uploaded_files = st.file_uploader(
        "Upload your documents (.pdf or .txt):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # FAISS index and metadata paths (hidden)
    faiss_index_path = "faiss_index.index"
    metadata_path = "metadata.json"

    # Initialize Vertex AI model
    model = ChatVertexAI(model="gemini-1.5-flash", project_id="mymlproject-444721")

    # Check if FAISS index exists and load or create
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        st.write("FAISS index exists. Loading FAISS index...")
        metadata = load_metadata(metadata_path)
        index = load_faiss_index(faiss_index_path)
    else:
        st.write("No FAISS index found. Please upload documents to create the index.")
        metadata, index = None, None

    # Update FAISS index and metadata if new files are uploaded
    new_index, new_metadata = update_index_with_new_files(
        uploaded_files, faiss_index_path, metadata_path, embeddings
    )
    if new_index and new_metadata:
        index = new_index
        metadata = new_metadata

    # Query System
    query = st.text_input("Enter your question:")

    if query:
        if index and metadata:
            search_similar_docs_with_faiss_and_generate_answer(query, index, metadata, embeddings, model)
        else:
            st.write("No documents available for querying. Please upload files.")

if __name__ == "__main__":
    main()

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

# Optimized parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Lower for more factual responses
            max_tokens=1000
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading data from URLs with proper headers..."
    try:
        # Use WebBaseLoader with custom headers to avoid blocking
        loader = WebBaseLoader(
            web_paths=urls,
            header_template={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        documents = loader.load()
    except Exception as e:
        yield f"Error loading URLs: {str(e)}"
        return

    if not documents:
        yield "Error: No documents loaded from URLs."
        return

    total_chars = sum(len(doc.page_content) for doc in documents)
    yield f"✓ Loaded {len(documents)} documents ({total_chars:,} characters)..."

    yield "Splitting text into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    docs = splitter.split_documents(documents)

    if not docs:
        yield "Error: No chunks created from documents."
        return

    yield f"✓ Created {len(docs)} text chunks..."

    yield "Generating and storing embeddings (this may take a moment)..."
    try:
        ids = [str(uuid4()) for _ in docs]
        vector_store.add_documents(documents=docs, ids=ids)

        # Verify storage
        collection = vector_store._collection
        count = collection.count()
        yield f"✓ SUCCESS! Vector store ready with {count} chunks stored."

    except Exception as e:
        yield f"✗ Error storing embeddings: {str(e)}"


def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector database not initialized. Process URLs first.")

    # Check if vector store has data
    try:
        count = vector_store._collection.count()
        print(f"DEBUG: Vector store has {count} documents")
        if count == 0:
            return "Vector store is empty. Please process URLs first.", ""
    except Exception as e:
        print(f"DEBUG: Error checking collection: {e}")

    # Retrieve relevant documents directly
    try:
        # Use similarity search with score
        docs_with_scores = vector_store.similarity_search_with_score(query, k=6)

        print(f"DEBUG: Found {len(docs_with_scores)} documents")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"DEBUG: Doc {i + 1} - Score: {score:.4f} - Preview: {doc.page_content[:100]}")

        if not docs_with_scores:
            return "No relevant information found in the knowledge base.", ""

        # Filter by relevance score (lower is better for distance metrics)
        # Adjusted threshold - be more lenient
        relevant_docs = [doc for doc, score in docs_with_scores if score < 2.5]

        if not relevant_docs:
            print("DEBUG: No documents passed score threshold, using top results")
            relevant_docs = [doc for doc, _ in docs_with_scores[:4]]  # Take top 4 anyway

    except Exception as e:
        print(f"DEBUG: Error during retrieval: {e}")
        return f"Error retrieving documents: {str(e)}", ""

    # Build context from retrieved documents
    context_parts = []
    sources = set()

    for i, doc in enumerate(relevant_docs[:5], 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content}\n")
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            sources.add(doc.metadata['source'])

    context = "\n".join(context_parts)

    # Create a focused prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above
- Be specific and detailed in your answer
- If the context contains relevant information, provide a comprehensive answer
- If you cannot find the answer in the context, say "I cannot find this information in the provided sources"
- Do not make up or infer information not present in the context

ANSWER:"""

    # Generate answer using LLM
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        # Format sources
        formatted_sources = "\n".join([f"- {src}" for src in sources]) if sources else "No sources cited"

        return answer.strip(), formatted_sources

    except Exception as e:
        print(f"DEBUG: Error generating answer: {e}")
        return f"Error generating answer: {str(e)}", ""
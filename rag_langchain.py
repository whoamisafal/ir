from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

class RAGSystem:
    def __init__(self, gemini_api_key):
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        # Initialize Embeddings
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        self.qdrant_client = QdrantClient(
            host='localhost',
            port=6333
        )
        
        self.collection_name = "documents1"
        self.vector_store = None
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Create the Qdrant collection if it does not exist yet"""
        existing_collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing_collections:
            print(f"⚡ Collection '{self.collection_name}' not found. Creating it...")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # <-- set embedding size manually
            )
            print(f" Collection '{self.collection_name}' created.")
        else:
            print(f" Collection '{self.collection_name}' already exists.")


    def _init_vector_store(self):
        """Ensures the vector store is connected to the local Qdrant instance"""
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )

    def add_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Adds documents to the Qdrant vector store.
        documents: list of langchain Document objects
        """
        if not documents:
            return 0

        self._init_vector_store()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)

        # Add chunks to vector store
        self.vector_store.add_documents(chunks)
        print(f" Added {len(chunks)} chunks to Qdrant collection '{self.collection_name}'")
        return len(chunks)

    def load_and_store(self, file_paths):
        """Helper to load and actually save docs to Qdrant if needed"""
        documents = []
        for path in file_paths:
            loader = PyPDFLoader(path) if path.endswith('.pdf') else TextLoader(path)
            documents.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        # Create/Update collection
        self.vector_store = QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name,
        )
        print(f"Saved {len(chunks)} chunks to Qdrant.")

    def create_rag_chain(self):
        # Ensure we are connected to Qdrant
        self._init_vector_store()
        
        system_prompt = (
            "You are a helpful and respectful assistant specialized in document summarization. "
            "Your task is to provide a clear, concise summary of the retrieved context below "
            "to answer the user's request. If the context does not contain the answer, "
            "please politely state that the information is not available. "
            "Please limit your response to a maximum of three sentences.\n\n"
            "Retrieved Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create the chains
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retriever with k=4
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        return create_retrieval_chain(retriever, question_answer_chain)
    
    def query(self, question):
        rag_chain = self.create_rag_chain()
        result = rag_chain.invoke({"input": question})
        return {
            "answer": result["answer"],
            "context": result["context"]
        }
            
    def qdrant_query_unique(self, question, top_k=5, fetch_k=30):
        """
        fetch_k: how many raw chunks to fetch from Qdrant before deduplication
        top_k: final number of unique documents (URLs)
        """

        embedding = self.embeddings.embed_query(question)
        
        # Qdrant query_points returns a QueryResponse object or a tuple 
        # based on the specific client version/method used.
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=fetch_k,
            with_payload=True
        )

        # UNPACKING: If response is ('points', [...]), we take index 1.
        # Otherwise, we check if it has a .points attribute.
        if isinstance(response, tuple):
            points = response[1]
        elif hasattr(response, 'points'):
            points = response.points
        else:
            points = response

        unique_results = []
        seen_urls = set()

        for r in points:
            payload = r.payload or {}
            metadata = payload.get("metadata", {})
            
            # ROBUS URL EXTRACTION: Handles 'url', 'url ', or 'url\xa0'
            url = None
            for k, v in metadata.items():
                if k.strip() == "url":
                    url = v
                    break

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            unique_results.append({
                "url": url,
                "title": url.split('/')[-1] if url else "Untitled", # Useful for the frontend
                "description": payload.get("page_content", ""),
                "visible_text": payload.get("page_content", ""), # Match your HTML requirement
                "score": float(r.score)
            })

            if len(unique_results) >= top_k:
                break

        return unique_results
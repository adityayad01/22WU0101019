import pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class PineconeHandler:
    def __init__(self, index_name="aditya"):
        self.index_name = index_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        
    def initialize(self):
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
            print("Pinecone initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            return False
    
    def add_knowledge(self, text, metadata=None):
        try:
            embedding = self.embedding_model.encode(text).tolist()
            doc_id = f"doc_{hash(text)}"
            
            self.index.upsert(
                vectors=[(doc_id, embedding, metadata or {"text": text})]
            )
            return True
        except Exception as e:
            print(f"Error adding knowledge: {e}")
            return False
    
    def search_knowledge(self, query, top_k=3):
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            contexts = []
            for match in results['matches']:
                if match['score'] > 0.5:
                    contexts.append(match['metadata'].get('text', ''))
            
            return "\n".join(contexts)
        except Exception as e:
            print(f"Error searching: {e}")
            return ""
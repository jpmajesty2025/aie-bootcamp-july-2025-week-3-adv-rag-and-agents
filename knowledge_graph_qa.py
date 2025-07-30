"""
Knowledge Graph Construction and Q&A using LangChain LLMGraphTransformer

This script demonstrates how to:
1. Extract structured information from unstructured text using LLMGraphTransformer
2. Store the information in a Neo4j graph database
3. Store text embeddings using LangChain's Neo4jVector for semantic search
4. Perform Q&A queries on the knowledge graph using Cypher and vector similarity

Required packages:
pip install langchain langchain-community langchain-openai langchain-experimental neo4j numpy
"""

import os
import getpass
import numpy as np
from typing import List, Optional, Dict, Any

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import GraphCypherQAChain
from langchain_text_splitters import CharacterTextSplitter


class KnowledgeGraphBuilder:
    """
    A class to build and query knowledge graphs using LLM Graph Transformer with Neo4j
    Includes LangChain Neo4jVector integration for embedding storage and semantic search
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o",
                 embedding_model: str = "text-embedding-3-small",
                 index_name: str = "document_embeddings",
                 keyword_index_name: str = "document_keywords"):
        """
        Initialize the Knowledge Graph Builder
        
        Args:
            openai_api_key: OpenAI API key (optional if set as env var)
            model_name: OpenAI model name to use
            embedding_model: OpenAI embedding model name
            index_name: Name for the vector index
            keyword_index_name: Name for the keyword index (for hybrid search)
            
        Environment Variables Required:
            NEO4J_URI: Neo4j database URI (e.g., bolt://localhost:7687)
            NEO4J_USERNAME: Neo4j username
            NEO4J_PASSWORD: Neo4j password
            OPENAI_API_KEY: OpenAI API key (optional if passed as parameter)
        """
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        
        # Initialize the LLM and embeddings
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Store index names
        self.index_name = index_name
        self.keyword_index_name = keyword_index_name
        
        # Initialize Neo4j graph database connection from environment variables
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME") 
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            missing_vars = []
            if not neo4j_uri: missing_vars.append("NEO4J_URI")
            if not neo4j_username: missing_vars.append("NEO4J_USERNAME")
            if not neo4j_password: missing_vars.append("NEO4J_PASSWORD")
            
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Store Neo4j connection details
        self.neo4j_url = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        try:
            self.graph_db = Neo4jGraph(enhanced_schema=True)
            print("✅ Connected to Neo4j database successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("Please check your Neo4j environment variables and database connection.")
            raise
        
        # Initialize the LLM Graph Transformer with schema
        self.llm_transformer: Optional[LLMGraphTransformer] = None
        self.setup_transformer()
        
        # Initialize Neo4j Vector Store (will be created when first used)
        self.vector_store: Optional[Neo4jVector] = None
        
    def setup_transformer(self, 
                         allowed_nodes: Optional[List[str]] = None,
                         allowed_relationships: Optional[List[str]] = None,
                         node_properties: bool = True):
        """
        Set up the LLM Graph Transformer with optional schema constraints
        
        Args:
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types  
            node_properties: Whether to extract node properties
        """
        # Default schema if none provided
        if allowed_nodes is None:
            allowed_nodes = ["Person", "Organization", "Location", "Event", "Concept", "Award", "Field"]
            
        if allowed_relationships is None:
            allowed_relationships = [
                "WORKS_AT",
                "BORN_IN",
                "LIVES_IN",
                "MARRIED_TO",
                "WON",
                "EXPERT_IN",
                "LOCATED_IN",
                "OCCURRED_IN",
                "PARTICIPATED_IN",
                "FOUNDED",
            ]
        
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=node_properties,
            strict_mode=True # Ensure adherence to schema
        )
        
        print(f"LLM Graph Transformer initialized with {len(allowed_nodes)} node types and {len(allowed_relationships)} relationship types")

    def create_vector_store_from_documents(self, 
                                         documents: List[Document], 
                                         search_type: str = "vector") -> Neo4jVector:
        """
        Create Neo4j vector store from documents using LangChain's Neo4jVector
        
        Args:
            documents: List of LangChain Document objects
            search_type: Type of search - "vector", "hybrid", or "keyword"
            
        Returns:
            Neo4jVector instance
        """
        try:
            print(f"Creating Neo4j vector store with {len(documents)} documents...")
            
            # Use LangChain's Neo4jVector.from_documents method
            vector_store = Neo4jVector.from_documents(
                documents,
                embedding=self.embeddings,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                keyword_index_name=self.keyword_index_name,
                search_type=search_type
            )
            
            print(f"✅ Neo4j vector store created successfully with search type: {search_type}")
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise

    def load_existing_vector_store(self, search_type: str = "vector") -> Neo4jVector:
        """
        Load existing Neo4j vector store using LangChain's Neo4jVector
        
        Args:
            search_type: Type of search - "vector", "hybrid", or "keyword"
            
        Returns:
            Neo4jVector instance
        """
        try:
            print(f"Loading existing Neo4j vector store...")
            
            kwargs = {
                "embedding": self.embeddings,
                "url": self.neo4j_url,
                "username": self.neo4j_username,
                "password": self.neo4j_password,
                "index_name": self.index_name,
                "search_type": search_type
            }
            
            # Add keyword index name for hybrid search
            if search_type == "hybrid":
                kwargs["keyword_index_name"] = self.keyword_index_name
            
            vector_store = Neo4jVector.from_existing_index(**kwargs)
            
            print(f"✅ Existing Neo4j vector store loaded successfully")
            return vector_store
            
        except Exception as e:
            print(f"❌ Error loading existing vector store: {e}")
            raise

    def store_text_embeddings(self, 
                            texts: List[str], 
                            metadata: Optional[List[Dict[str, Any]]] = None,
                            search_type: str = "vector",
                            chunk_size: int = 1000,
                            chunk_overlap: int = 200) -> Neo4jVector:
        """
        Store text chunks with their embeddings using LangChain's Neo4jVector
        
        Args:
            texts: List of text strings to store
            metadata: Optional metadata for each text chunk
            search_type: Type of search - "vector", "hybrid", or "keyword"
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Neo4jVector instance
        """
        try:
            # Split texts into chunks for better retrieval
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n"
            )
            
            # Create Document objects
            documents = []
            for i, text in enumerate(texts):
                # Split individual text into chunks
                chunks = text_splitter.split_text(text)
                
                for j, chunk in enumerate(chunks):
                    doc_metadata = metadata[i].copy() if metadata and i < len(metadata) else {}
                    doc_metadata.update({
                        "source_doc_id": i,
                        "chunk_id": j,
                        "total_chunks": len(chunks)
                    })
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    ))
            
            print(f"Created {len(documents)} document chunks from {len(texts)} texts")
            
            # Create vector store
            self.vector_store = self.create_vector_store_from_documents(documents, search_type)
            
            print(f"✅ Stored {len(documents)} document chunks with embeddings")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ Error storing text embeddings: {e}")
            raise

    def add_documents_to_vector_store(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add new documents to existing vector store
        
        Args:
            texts: List of text strings to add
            metadata: Optional metadata for each text chunk
        """
        try:
            if self.vector_store is None:
                print("⚠️  No vector store exists. Creating new one...")
                self.store_text_embeddings(texts, metadata)
                return
            
            # Create Document objects
            documents = []
            for i, text in enumerate(texts):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                documents.append(Document(page_content=text, metadata=doc_metadata))
            
            # Add documents to existing vector store
            doc_ids = self.vector_store.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            raise

    def similarity_search(self, 
                         query_text: str, 
                         k: int = 5, 
                         filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform similarity search using LangChain's Neo4jVector
        
        Args:
            query_text: Text to search for similar documents
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            if self.vector_store is None:
                print("❌ No vector store available. Please create one first.")
                return []
            
            # Perform similarity search
            if filter:
                results = self.vector_store.similarity_search(query_text, k=k, filter=filter)
            else:
                results = self.vector_store.similarity_search(query_text, k=k)
            
            print(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []

    def similarity_search_with_score(self, 
                                   query_text: str, 
                                   k: int = 5,
                                   filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Perform similarity search with scores using LangChain's Neo4jVector
        
        Args:
            query_text: Text to search for similar documents
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if self.vector_store is None:
                print("❌ No vector store available. Please create one first.")
                return []
            
            # Perform similarity search with scores
            if filter:
                results = self.vector_store.similarity_search_with_score(query_text, k=k, filter=filter)
            else:
                results = self.vector_store.similarity_search_with_score(query_text, k=k)
            
            print(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            print(f"❌ Error in similarity search with score: {e}")
            return []

    def extract_graph_from_text(self, texts: List[str]) -> List:
        """
        Extract graph documents from text using LLMGraphTransformer
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of graph documents
        """
        if self.llm_transformer is None:
            raise ValueError("LLM Transformer not initialized. Call setup_transformer() first.")
            
        documents = [Document(page_content=text) for text in texts]
        
        print(f"Processing {len(documents)} documents...")
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        
        print(f"Extracted {len(graph_documents)} graph documents")
        for i, doc in enumerate(graph_documents):
            print(f"Document {i+1}: {len(doc.nodes)} nodes, {len(doc.relationships)} relationships")
            
        return graph_documents

    def store_graph_documents(self, graph_documents: List, include_source: bool = True):
        """
        Store graph documents in the Neo4j database
        
        Args:
            graph_documents: List of graph documents to store
            include_source: Whether to include source document information
        """
        try:
            self.graph_db.add_graph_documents(
                graph_documents, 
                baseEntityLabel=True,
                include_source=include_source
            )
            print("✅ Graph documents stored in Neo4j database successfully")
        except Exception as e:
            print(f"❌ Failed to store graph documents: {e}")
            raise

    def process_and_store_all(self, 
                            texts: List[str], 
                            metadata: Optional[List[Dict[str, Any]]] = None,
                            search_type: str = "hybrid"):
        """
        Process texts to extract both graph data and embeddings, then store both
        
        Args:
            texts: List of text strings to process
            metadata: Optional metadata for each text chunk
            search_type: Type of search - "vector", "hybrid", or "keyword"
        """
        print("=== Processing texts for both graph extraction and embedding storage ===")
        
        # 1. Extract and store knowledge graph
        print("\n1. Extracting knowledge graph...")
        graph_documents = self.extract_graph_from_text(texts)
        
        print("2. Storing graph documents...")
        self.store_graph_documents(graph_documents)
        
        # 2. Store text embeddings using LangChain Neo4jVector
        print("3. Storing text embeddings using LangChain Neo4jVector...")
        self.store_text_embeddings(texts, metadata, search_type)
        
        print("✅ All processing and storage completed!")

    def hybrid_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Perform hybrid search using both graph relationships and vector similarity
        
        Args:
            query: Search query
            k: Number of results to return from similarity search
            
        Returns:
            Combined results from graph query and similarity search
        """
        print(f"Performing hybrid search for: '{query}'")
        
        # 1. Graph-based query
        print("1. Querying knowledge graph...")
        graph_result = self.query_graph(query)
        
        # 2. Vector similarity search
        print("2. Performing vector similarity search...")
        similar_docs = self.similarity_search_with_score(query, k=k)
        
        return {
            "graph_answer": graph_result,
            "similar_documents": similar_docs,
            "query": query
        }

    def generate_comprehensive_response(self, 
                                      query: str, 
                                      graph_result: str, 
                                      similar_documents: List[tuple],
                                      max_docs: int = 3) -> str:
        """
        Generate a comprehensive response using LLM with hybrid search results
        
        Args:
            query: Original user query
            graph_result: Result from graph database query
            similar_documents: List of (document, score) tuples from vector search
            max_docs: Maximum number of similar documents to include
            
        Returns:
            Comprehensive LLM-generated response
        """
        try:
            # Prepare context from similar documents
            context_docs = []
            for i, (doc, score) in enumerate(similar_documents[:max_docs]):
                context_docs.append(f"Document {i+1} (similarity: {score:.3f}):\n{doc.page_content}\nMetadata: {doc.metadata}\n")
            
            context_text = "\n---\n".join(context_docs) if context_docs else "No similar documents found."
            
            # Create comprehensive prompt template
            prompt_template = PromptTemplate(
                input_variables=["query", "graph_result", "context_documents"],
                template="""You are an AI assistant that provides comprehensive answers by combining structured knowledge from a graph database with relevant document context.

User Query: {query}

GRAPH DATABASE RESULT:
{graph_result}

RELEVANT DOCUMENTS FROM VECTOR SEARCH:
{context_documents}

Instructions:
1. Analyze both the graph database result and the relevant documents
2. Provide a comprehensive answer that synthesizes information from both sources
3. If the graph result provides direct answers, use that as the primary source
4. Use the relevant documents to add context, details, or supporting information
5. If there are contradictions, note them and explain which source might be more reliable
6. Cite specific information when possible (e.g., "According to the graph database..." or "From the documents...")
7. If neither source provides sufficient information, clearly state what information is missing

Comprehensive Answer:"""
            )
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                query=query,
                graph_result=graph_result,
                context_documents=context_text
            )
            
            # Generate response using LLM
            print("3. Generating comprehensive response with LLM...")
            response = self.llm.invoke(formatted_prompt)
            
            # Ensure we return a string
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
            
        except Exception as e:
            print(f"❌ Error generating comprehensive response: {e}")
            return f"Error generating response: {str(e)}"

    def hybrid_search_with_llm_response(self, 
                                       query: str, 
                                       k: int = 5, 
                                       max_docs: int = 3) -> Dict[str, Any]:
        """
        Perform hybrid search and generate a comprehensive LLM response
        
        Args:
            query: Search query
            k: Number of results to return from similarity search
            max_docs: Maximum number of similar documents to include in LLM context
            
        Returns:
            Combined results with LLM-generated comprehensive response
        """
        # Perform hybrid search
        hybrid_results = self.hybrid_search(query, k)
        
        # Generate comprehensive response
        comprehensive_response = self.generate_comprehensive_response(
            query=query,
            graph_result=hybrid_results["graph_answer"],
            similar_documents=hybrid_results["similar_documents"],
            max_docs=max_docs
        )
        
        # Add comprehensive response to results
        hybrid_results["comprehensive_response"] = comprehensive_response
        
        return hybrid_results

    def get_vector_store_as_retriever(self, 
                                    search_type: str = "similarity",
                                    search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get vector store as a LangChain retriever
        
        Args:
            search_type: Type of search ("similarity", "similarity_score_threshold", "mmr")
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain retriever
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call store_text_embeddings() first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def query_graph(self, query: str) -> str:
        """
        Query the Neo4j graph using natural language
        
        Args:
            query: Natural language query
            
        Returns:
            Answer from the graph using Cypher
        """
        try:
            qa_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph_db,
                verbose=True,
                allow_dangerous_requests=True
            )
            
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            print(f"❌ Error querying graph: {e}")
            return f"Sorry, I couldn't process your query: {str(e)}"

    def get_graph_statistics(self) -> dict:
        """
        Get basic statistics about the Neo4j knowledge graph including embeddings
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            stats = {}
            stats["total_nodes"] = self.graph_db.query("MATCH (n) RETURN count(n) as count")[0]["count"]
            stats["total_relationships"] = self.graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
            
            # Check for vector store nodes
            vector_nodes = self.graph_db.query("MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as count")[0]["count"]
            stats["nodes_with_embeddings"] = vector_nodes
            
            # Get node types
            node_types = self.graph_db.query("MATCH (n) RETURN DISTINCT labels(n) as labels")
            stats["node_types"] = [item["labels"][0] for item in node_types if item["labels"]]
            
            # Get relationship types  
            rel_types = self.graph_db.query("MATCH ()-[r]->() RETURN DISTINCT type(r) as type")
            stats["relationship_types"] = [item["type"] for item in rel_types]
            
            return stats
        except Exception as e:
            print(f"❌ Error getting graph statistics: {e}")
            return {"error": str(e)}


def main():
    """
    Demo function showing how to use the Knowledge Graph Builder with LangChain Neo4jVector
    """
    
    # Sample text data about famous scientists
    sample_texts = [
        """
        Marie Curie, born in 1867 in Warsaw, Poland, was a Polish and naturalized-French physicist and chemist. 
        She conducted pioneering research on radioactivity and was the first woman to win a Nobel Prize. 
        She was also the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
        Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize.
        She worked at the University of Paris and became the first woman professor there in 1906.
        """,
        
        """
        Albert Einstein was born in 1879 in Ulm, Germany. He was a German-born theoretical physicist who developed the theory of relativity.
        He won the Nobel Prize in Physics in 1921 for his services to theoretical physics, particularly for his discovery of the law of the photoelectric effect.
        Einstein worked at Princeton University and became a prominent figure in the field of theoretical physics.
        He is widely regarded as one of the greatest physicists of all time.
        """,
        
        """
        Isaac Newton was born in 1642 in Woolsthorpe, England. He was an English mathematician, physicist, astronomer, and author.
        Newton is widely recognized as one of the most influential scientists of all time and a key figure in the scientific revolution.
        He formulated the laws of motion and universal gravitation. Newton worked at Cambridge University as a professor.
        His work laid the foundation for classical mechanics.
        """
    ]
    
    # Sample metadata for each text
    sample_metadata = [
        {"scientist": "Marie Curie", "field": "Physics/Chemistry", "era": "19th/20th century", "country": "Poland/France"},
        {"scientist": "Albert Einstein", "field": "Theoretical Physics", "era": "20th century", "country": "Germany/USA"},
        {"scientist": "Isaac Newton", "field": "Mathematics/Physics", "era": "17th century", "country": "England"}
    ]
    
    print("=== Knowledge Graph Builder with LangChain Neo4jVector Demo ===\n")
    
    try:
        # Initialize the builder (reads Neo4j credentials from environment variables)
        kg_builder = KnowledgeGraphBuilder()
        
        # Process and store everything (graph + embeddings using hybrid search)
        print("\n1. Processing texts for both graph extraction and embedding storage...")
        kg_builder.process_and_store_all(sample_texts, sample_metadata, search_type="hybrid")
        
        # Get graph statistics
        print("\n2. Graph Statistics:")
        stats = kg_builder.get_graph_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test hybrid search with LLM response
        print("\n Testing hybrid search with comprehensive LLM responses...")
        hybrid_queries = [
            "Who won the Nobel Prize and what did they contribute to science?",
            "Which scientists worked at universities and what were their main achievements?",
            "Tell me about the scientific contributions of physicists in the 20th century"
        ]
        
        for query in hybrid_queries:
            print(f"\n{'='*80}")
            print(f"🔍 Query: {query}")
            print('='*80)
            
            results = kg_builder.hybrid_search_with_llm_response(query, k=4, max_docs=3)
            
            # print(f"\n📊 GRAPH DATABASE RESULT:")
            # print(f"{results['graph_answer']}")
            
            # print(f"\n📄 SIMILAR DOCUMENTS FOUND: {len(results['similar_documents'])}")
            # for i, (doc, score) in enumerate(results['similar_documents'][:2]):  # Show top 2
            #     print(f"  {i+1}. Score: {score:.3f} - {doc.page_content[:80]}...")
            
            print(f"\n🤖 COMPREHENSIVE LLM RESPONSE:")
            print(f"{results['comprehensive_response']}")
            print("-" * 80)
        
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check your Neo4j connection and try again.")


if __name__ == "__main__":
    main() 
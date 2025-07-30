# Knowledge Graph Construction and Q&A with LangChain

This project demonstrates how to use LangChain's `LLMGraphTransformer` to extract structured information from unstructured text, build a knowledge graph in Neo4j, and perform Q&A queries using Cypher.

## Features

- **Graph Extraction**: Convert unstructured text into structured knowledge graphs using LLMs
- **Neo4j Storage**: Store graphs in Neo4j database with full Cypher query support
- **Schema-Controlled Extraction**: Define allowed node types and relationships for consistent extraction
- **Natural Language Q&A**: Query the knowledge graph using natural language (converted to Cypher)
- **Property Extraction**: Extract node and relationship properties from text
- **Graph Visualization**: Use Neo4j Browser for powerful graph visualization

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   
   **OpenAI API Key:**
   - Get an API key from [OpenAI](https://platform.openai.com/api-keys)
   - Set as environment variable: `export OPENAI_API_KEY=your_key_here`
   
   **Neo4j Database:**
   - **Option 1**: Install [Neo4j Desktop](https://neo4j.com/download/) and create a local database
   - **Option 2**: Use [Neo4j Aura](https://neo4j.com/cloud/aura/) (free cloud database)
   - Set Neo4j environment variables:
     ```bash
     export NEO4J_URI=bolt://localhost:7687
     export NEO4J_USERNAME=neo4j
     export NEO4J_PASSWORD=your_password
     ```
   
   **Alternative: Use a .env file**
   ```bash
   # Create a .env file in your project directory with:
   OPENAI_API_KEY=your_openai_api_key_here
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password_here
   ```
   
   Then load it in your Python script:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load .env file
   
   from knowledge_graph_qa import KnowledgeGraphBuilder
   kg_builder = KnowledgeGraphBuilder()
   ```

## Usage

### Basic Usage

```python
from knowledge_graph_qa import KnowledgeGraphBuilder

# Initialize (reads Neo4j credentials from environment variables)
kg_builder = KnowledgeGraphBuilder()

# Extract graph from your text
texts = ["Your text documents here..."]
graph_documents = kg_builder.extract_graph_from_text(texts)

# Store the graph in Neo4j
kg_builder.store_graph_documents(graph_documents)

# Query the graph using natural language
answer = kg_builder.query_graph("Who won the Nobel Prize?")
print(answer)
```

### Run the Demo

First, set your environment variables, then run the script:

```bash
# Set environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
export OPENAI_API_KEY=your_openai_key

# Run the demo
python knowledge_graph_qa.py
```

The script will:
1. Extract a knowledge graph from sample text about Marie Curie, Einstein, and Newton
2. Store it in your Neo4j database
3. Display graph statistics
4. Answer several example questions using Cypher queries
5. Provide instructions for viewing the graph in Neo4j Browser

## Customization

### Custom Schema

You can define your own node types and relationships:

```python
# Initialize (reads credentials from environment variables)
kg_builder = KnowledgeGraphBuilder()

# Set up custom schema
kg_builder.setup_transformer(
    allowed_nodes=["Person", "Company", "Technology"],
    allowed_relationships=[
        ("Person", "WORKS_AT", "Company"),
        ("Person", "USES", "Technology"),
        ("Company", "DEVELOPS", "Technology")
    ]
)
```

### Custom Queries

Ask any natural language question about your knowledge graph:

```python
queries = [
    "Who are the people in the graph?",
    "What companies are mentioned?",
    "What relationships exist between entities?",
    "Which person worked at which organization?"
]

for query in queries:
    answer = kg_builder.query_graph(query)
    print(f"Q: {query}")
    print(f"A: {answer}\n")
```

## Architecture

The system consists of several key components:

1. **LLMGraphTransformer**: Extracts entities and relationships from text using LLMs
2. **Neo4j Storage**: Stores structured graph data in Neo4j database
3. **Cypher Query Engine**: Converts natural language to Cypher queries for graph traversal
4. **Visualization**: Use Neo4j Browser for interactive graph exploration

## Example Output

When you run the demo, you'll see output like:

```
=== Knowledge Graph Builder Demo ===

✅ Connected to Neo4j database successfully
LLM Graph Transformer initialized with 7 node types and 10 relationship types

1. Extracting knowledge graph from text...
Processing 3 documents...
Extracted 3 graph documents
Document 1: 5 nodes, 4 relationships
Document 2: 4 nodes, 3 relationships
Document 3: 4 nodes, 2 relationships

2. Storing graph in Neo4j database...
✅ Graph documents stored in Neo4j database successfully

3. Graph Statistics:
   total_nodes: 11
   total_relationships: 9
   node_types: ['Person', 'Location', 'Organization', 'Award', 'Field']
   relationship_types: ['BORN_IN', 'WON', 'MARRIED_TO', 'WORKS_AT', 'EXPERT_IN']

Query: Who won the Nobel Prize?
Answer: Based on the knowledge graph, Marie Curie, Albert Einstein, and Pierre Curie won the Nobel Prize. Marie Curie was the first woman to win a Nobel Prize and won it twice in two different scientific fields. Albert Einstein won the Nobel Prize in Physics in 1921. Pierre Curie was a co-winner with Marie Curie.

✅ Demo completed! You can now:
   - View your graph in Neo4j Browser
   - Run additional queries using kg_builder.query_graph()
   - Add more documents using kg_builder.extract_graph_from_text()
```

## Requirements

- Python 3.8+
- OpenAI API key
- Neo4j database (local or cloud)

## Notes

- The quality of extraction depends on the LLM model used (GPT-4 recommended)
- Neo4j provides powerful graph querying capabilities via Cypher
- Use Neo4j Browser for interactive graph visualization and exploration
- The schema definition greatly improves extraction consistency
- GraphCypherQAChain automatically converts natural language to Cypher queries

## Contributing

Feel free to enhance the project by:
- Adding more sophisticated query processing
- Implementing additional graph storage backends
- Improving the visualization capabilities
- Adding more extraction schemas for different domains 
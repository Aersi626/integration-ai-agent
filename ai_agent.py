import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import faiss # Although not directly used in the provided snippet, it's a dependency for FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, END # Added END for clarity, though not strictly needed if only one path
from typing import List, TypedDict
from langfuse.langchain import CallbackHandler
from tqdm import tqdm # For progress indicators

# --- Langfuse Setup ---
# IMPORTANT: Replace these with your actual keys from http://localhost:3000/settings
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-4775e411-197e-4e81-8041-cf6325ab3a18"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-f8a28219-7a5c-4274-ae55-9d6a57a681fb"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_RELEASE"] = "local-rag-app"
os.environ["LANGFUSE_DEBUG"] = "TRUE"

# Initialize the Langfuse CallbackHandler globally in this module
langfuse_handler = CallbackHandler()

# --- End Langfuse Setup ---

# --- Global Variables for RAG Components (within this module) ---
# These will be initialized by initialize_rag_pipeline()
_vector_store = None
_retriever = None
_prompt_template = None
_llm = None
_app_langgraph = None # The compiled LangGraph application

# Define the GraphState TypedDict to represent the state of our graph.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's implicit query for retrieval (instructions + original YAML).
        context: List of retrieved documents (as strings) from the vector store.
        original_yaml_string: The initial YAML string provided by the user.
        instructions: The instructions for modifying the YAML.
        yaml_output: The final generated (modified) YAML string from the LLM.
    """
    question: str
    context: List[str]
    original_yaml_string: str
    instructions: str
    yaml_output: str

# --- LangGraph Nodes (Functions) - Moved outside main() ---
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieves relevant documents from the FAISS vector store based on the 'question'
    in the current graph state.
    """
    print("---RETRIEVING DOCUMENTS (in ai_agent.py)---")
    question = state["question"]
    # Access the module-level retriever
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call initialize_rag_pipeline first.")
    retrieved_docs: List[Document] = _retriever.invoke(question) 
    
    return {
        "context": [doc.page_content for doc in retrieved_docs],
        "question": question,
        "original_yaml_string": state["original_yaml_string"],
        "instructions": state["instructions"],
        "yaml_output": "" 
    }

def generate(state: GraphState) -> GraphState:
    """
    Generates the modified YAML using the Ollama LLM,
    incorporating the retrieved context and instructions.
    """
    print("---GENERATING YAML (in ai_agent.py)---")
    question = state["question"] 
    context = state["context"] 
    original_yaml_string = state["original_yaml_string"] 
    instructions = state["instructions"] 

    context_str = "\n\n".join(context)

    # Access the module-level prompt_template and llm
    if _prompt_template is None or _llm is None:
        raise RuntimeError("LLM or PromptTemplate not initialized. Call initialize_rag_pipeline first.")

    formatted_prompt = _prompt_template.format(
        context=context_str,
        original_yaml_string=original_yaml_string,
        instructions=instructions
    )

    llm_response = _llm.invoke(formatted_prompt)

    return {
        "yaml_output": llm_response,
        "question": question,
        "context": context,
        "original_yaml_string": original_yaml_string,
        "instructions": instructions
    }

# --- Initialization Function for RAG Components ---
def initialize_rag_pipeline():
    """
    Initializes all RAG components (document loader, splitter, embeddings,
    vector store, LLM, prompt template, and LangGraph workflow).
    This function should be called once at application startup.
    """
    global _vector_store, _retriever, _prompt_template, _llm, _app_langgraph

    print("---Initializing RAG components (in ai_agent.py)---")

    # 1. Document loading and processing
    try:
        print("Loading documents from data/DSL_example.txt...")
        loader = TextLoader("./data/DSL_example.txt", encoding="utf-8")
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
    except FileNotFoundError:
        print("Error: 'data/DSL_example.txt' not found. Using dummy content for demonstration.")
        documents = [Document(page_content="Apache Camel is an open-source integration framework that allows you to quickly and easily integrate various systems consuming or producing data using the appropriate EIPs (Enterprise Integration Patterns). It supports many data formats and protocols. Transformations are a key part of integration, often done with `setBody` or `transform` steps.")]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    
    texts = []
    with tqdm(total=len(documents), desc="Splitting documents") as pbar:
        for doc in documents:
            texts.extend(text_splitter.split_documents([doc]))
            pbar.update(1)
    print(f"Split into {len(texts)} chunks.")

    # 2. Initialize Embeddings and FAISS Vector Store
    print("Initializing Ollama embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Embedding chunks and building FAISS index (this can be slow)...")
    _vector_store = FAISS.from_documents(texts, embeddings)
    _retriever = _vector_store.as_retriever()
    print("FAISS index built and retriever created.")

    # 3. Initialize LLM and Prompt Template
    print("Initializing Ollama LLM (codellama:7b)...")
    _llm = OllamaLLM(model="codellama:7b")
    _prompt_template = PromptTemplate.from_template('''
you are an expert in YAML files and apache camel integration.
Context:
{context}
Here is a YAML file that I want you to make change:
{original_yaml_string}
Can you make changes to this YAML file based on the following instructions? Be really careful with the syntax and structure of the YAML and indentation.
Instructions:
{instructions}
After making the change, just respond with the new YAML file in YAML format.
''')
    print("LLM and Prompt Template initialized.")

    # 4. Compile LangGraph workflow
    print("Compiling LangGraph workflow...")
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge("retrieve", "generate")
    workflow.set_entry_point("retrieve")
    workflow.set_finish_point("generate")
    _app_langgraph = workflow.compile()
    print("LangGraph workflow compiled.")
    print("---RAG components initialization complete (in ai_agent.py)---")

def invoke_agent(original_yaml: str, instructions: str):
    """
    Invokes the compiled LangGraph agent with the given YAML and instructions.
    Returns the transformed YAML output.
    """
    if _app_langgraph is None:
        raise RuntimeError("LangGraph agent not initialized. Call initialize_rag_pipeline first.")

    initial_question_for_retrieval = f"Instructions for YAML modification: {instructions}\nOriginal YAML: {original_yaml}"
    initial_state = {
        "question": initial_question_for_retrieval,
        "context": [],
        "original_yaml_string": original_yaml,
        "instructions": instructions,
        "yaml_output": ""
    }

    print("\n--- Invoking LangGraph Application ---")
    final_state = _app_langgraph.invoke(initial_state, config={"callbacks": [langfuse_handler]})
    print("--- LangGraph Application invoked successfully ---")

    # Ensure Langfuse data is flushed
    langfuse_handler.client.flush()
    print("--- Langfuse data flushed ---")

    return final_state.get('yaml_output', 'No output generated.')

# This block allows ai_agent.py to be run directly for testing purposes
if __name__ == "__main__":
    # Ensure 'data' directory exists for DSL_example.txt
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory.")
    
    # Create a dummy DSL_example.txt if it doesn't exist
    dsl_example_path = "./data/DSL_example.txt"
    if not os.path.exists(dsl_example_path):
        with open(dsl_example_path, "w", encoding="utf-8") as f:
            f.write("Apache Camel is an open-source integration framework that allows you to quickly and easily integrate various systems consuming or producing data using the appropriate EIPs (Enterprise Integration Patterns). It supports many data formats and protocols. Transformations are a key part of integration, often done with `setBody` or `transform` steps.")
        print(f"Created dummy {dsl_example_path} as it was not found.")

    initialize_rag_pipeline()
    
    # Example usage for direct testing
    example_yaml = """
- route:
    id: route-45c5
    nodePrefixId: route-a5b
    from:
      id: from-7f09
      uri: timer
      parameters:
        timerName: test-timer
        period: "2000"
        repeatCount: "10"
      steps:
        - log:
            id: log-3fc3
            message: "**** hello world ****"
            loggingLevel: INFO
    """
    example_instructions = "change the route id to \"my-updated-route-id\" and change the timer period to 1 seconds. Also, change the printed message to \"Hello from the updated Camel route!\"."
    
    transformed_output = invoke_agent(example_yaml, example_instructions)
    print("\n--- Transformed YAML (from direct ai_agent.py run) ---")
    print(transformed_output)

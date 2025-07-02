import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph
from typing import List, TypedDict
from langfuse.langchain import CallbackHandler

# IMPORTANT: Replace these with your actual keys from http://localhost:3000/settings
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-4775e411-197e-4e81-8041-cf6325ab3a18"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-f8a28219-7a5c-4274-ae55-9d6a57a681fb"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_RELEASE"] = "local-rag-app"
os.environ["LANGFUSE_DEBUG"] = "TRUE"

# Initialize the Langfuse CallbackHandler
# It will automatically pick up the environment variables you set above.
langfuse_handler = CallbackHandler()
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

def main():
    try:
        loader = TextLoader("./data/DSL_example.txt", encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print("Error: 'data/DSL_example.txt' not found. Please create the file with relevant content.")
        documents = [Document(page_content="Apache Camel is an open-source integration framework that allows you to quickly and easily integrate various systems consuming or producing data using the appropriate EIPs (Enterprise Integration Patterns).")]
    print("---LOADED TEXTS---")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    print(f"---SPLIT INTO {len(texts)} TEXT CHUNKS---")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("---INITIALIZED OLLAMA EMBEDDINGS---")
    
    vector_store = FAISS.from_documents(texts, embeddings)
    print("---CREATED FAISS VECTOR STORE---")
    
    retriever = vector_store.as_retriever()
    print("---CREATED RETRIEVER FROM VECTOR STORE---")
    
    instructions = '''change the route id to "my-updated-route-id" and change the timer period to 1 seconds.
    Also, change the printed message to "Hello from the updated Camel route!".'''

    yaml_string = """
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

    prompt_template = PromptTemplate.from_template('''
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

    llm = OllamaLLM(model="codellama:7b")
    print("---INITIALIZED OLLAMA LLM---")
    
    # Define Nodes (Steps) in your Graph
    def retrieve(state: GraphState) -> GraphState:
        """
        Retrieves relevant documents from the FAISS vector store based on the 'question' in the current graph state.
        """
        print("---RETRIEVING DOCUMENTS---")
        question = state["question"]
        retrieved_docs: List[Document] = retriever.invoke(question) 
        return {
            "context": [doc.page_content for doc in retrieved_docs],
            "question": question,
            "original_yaml_string": state["original_yaml_string"],
            "instructions": state["instructions"],
            "yaml_output": "" # Ensure yaml_output is reset or kept empty for this step
        }

    def generate(state: GraphState) -> GraphState:
        """
        Generates the modified YAML using the Ollama LLM, incorporating the retrieved context and instructions.
        """
        print("---GENERATING YAML---")
        question = state["question"]
        context = state["context"]
        original_yaml_string = state["original_yaml_string"]
        instructions = state["instructions"]

        # Combine the list of context strings into a single string for the prompt
        context_str = "\n\n".join(context)

        formatted_prompt = prompt_template.format(
            context=context_str,
            original_yaml_string=original_yaml_string,
            instructions=instructions
        )

        llm_response = llm.invoke(formatted_prompt)
        return {
            "yaml_output": llm_response,
            "question": question,
            "context": context,
            "original_yaml_string": original_yaml_string,
            "instructions": instructions
        }
    # Build the LangGraph Workflow
    # Initialize a StateGraph with the defined GraphState
    workflow = StateGraph(GraphState)
    # Add the 'retrieve' and 'generate' nodes to the workflow
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    # Define the flow (edges) between the nodes
    # After 'retrieve' node completes, the flow moves to the 'generate' node
    workflow.add_edge("retrieve", "generate")
    # Set the entry point of the graph (where the execution begins)
    workflow.set_entry_point("retrieve")
    # Set the finish point of the graph (where the execution ends)
    workflow.set_finish_point("generate")
    # Compile the graph into a runnable application
    app = workflow.compile()

    initial_question_for_retrieval = f"Instructions for YAML modification: {instructions}\nOriginal YAML: {yaml_string}"

    initial_state = {
        "question": initial_question_for_retrieval,
        "context": [],  # This will be populated by the 'retrieve' node
        "original_yaml_string": yaml_string,
        "instructions": instructions,
        "yaml_output": ""  # This will be populated by the 'generate' node
    }

    print("--- Invoking LangGraph Application ---")
    final_state = app.invoke(initial_state, config={"callbacks": [langfuse_handler]})

    print("\n--- Final Generated YAML ---")
    print(final_state['yaml_output'])

    print("--- Flushing Langfuse data ---")
    langfuse_handler.client.flush()
    print("--- Langfuse data flushed ---")

    # (Optional, but good for robust shutdown, especially in more complex apps)
    # langfuse_handler.client.shutdown()

if __name__ == "__main__":
    main()
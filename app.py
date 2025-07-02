import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time

# Import the core AI agent logic from ai_agent.py
from ai_agent import initialize_rag_pipeline, invoke_agent, langfuse_handler

# --- Flask Application Setup ---
flask_app = Flask(__name__)
CORS(flask_app)

# --- Frontend UI Route ---
@flask_app.route('/')
def index():
    """Serves the main HTML UI page from the templates folder."""
    return render_template('index.html')

# --- API Endpoint for YAML Transformation ---
@flask_app.route('/transform_yaml', methods=['POST'])
def transform_yaml_api():
    """
    API endpoint to receive original YAML and instructions,
    run the LangGraph transformation via ai_agent.py, and return the modified YAML.
    """
    data = request.json
    original_yaml = data.get('original_yaml', '')
    instructions = data.get('instructions', '')

    if not original_yaml or not instructions:
        return jsonify({"error": "Missing original_yaml or instructions"}), 400

    print("\n--- Received request for YAML transformation ---")
    try:
        # Call the invoke_agent function from ai_agent.py
        transformed_yaml = invoke_agent(original_yaml, instructions)
        return jsonify({"transformed_yaml": transformed_yaml})
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        # Ensure Langfuse data is flushed even on error
        langfuse_handler.client.flush()
        return jsonify({"error": f"An error occurred during transformation: {str(e)}"}), 500

# --- Run the Flask Application ---
if __name__ == '__main__':
    # Ensure 'data' directory exists for DSL_example.txt
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory.")
    
    dsl_example_path = "./data/DSL_example.txt"
    if not os.path.exists(dsl_example_path):
        with open(dsl_example_path, "w", encoding="utf-8") as f:
            f.write("Apache Camel is an open-source integration framework that allows you to quickly and easily integrate various systems consuming or producing data using the appropriate EIPs (Enterprise Integration Patterns). It supports many data formats and protocols. Transformations are a key part of integration, often done with `setBody` or `transform` steps.")
        print(f"Created dummy {dsl_example_path} as it was not found.")

    # Call the initialization function directly before running the app.
    # This ensures RAG components are set up once when the server starts.
    initialize_rag_pipeline()
    print("---Flask app has initialized RAG components---")

    # Run Flask app on port 5000 to avoid conflict with Langfuse UI (3000)
    flask_app.run(debug=True, port=5000)

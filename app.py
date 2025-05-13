from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.deepseek import DeepSeek  # New DeepSeek LLM integration
import os
import logging
from dotenv import load_dotenv, dotenv_values

app = Flask(__name__)
load_dotenv()
config = dotenv_values(".env")

CORS(app, origins=[config.get("CLIENT_URL", "*")])

# Set DeepSeek API key
DEEPSEEK_API_KEY = config.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    app.logger.error("DEEPSEEK_API_KEY not found in .env file or environment variables.")
    # Exit if API key is critical for startup
    raise ValueError("DeepSeek API key is required")

# Configure LlamaIndex settings with DeepSeek
Settings.llm = DeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.7
)

# Global variable for the index
index = None

def construct_index(directory_path):
    global index
    try:
        app.logger.info(f"Constructing index from directory: {directory_path}")
        documents = SimpleDirectoryReader(directory_path).load_data()
        if not documents:
            app.logger.error(f"No documents found in directory: {directory_path}")
            return None
        index = VectorStoreIndex.from_documents(documents)
        app.logger.info("Index constructed successfully.")
        return index
    except Exception as e:
        app.logger.error(f"Error constructing index: {e}")
        return None

def generate_response(input_text):
    global index
    if index is None:
        app.logger.error("Index is not loaded.")
        return "Error: Chatbot index is not available. Please try again later."
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(input_text)
        return str(response)
    except Exception as e:
        app.logger.error(f"Error generating response: {e}")
        return "Error: Could not generate a response."

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    input_text = request.json.get("input_text")
    if not input_text:
        return make_response(jsonify({"error": "input_text is required"}), 400)
    
    response_text = generate_response(input_text)
    return make_response(jsonify({"response": response_text}))

@app.route("/api/chatbot/reload", methods=["POST"])
def reload_index_route():
    app.logger.info("Reloading index via API call...")
    if construct_index("docs"):
        return make_response(jsonify({"message": "Index reloaded successfully"}), 200)
    else:
        return make_response(jsonify({"error": "Failed to reload index"}), 500)

@app.route("/api/chatbot/hello", methods=["GET"])
def hello():
    return make_response(jsonify({
        "success": True,
        "message": "Hello from your DeepSeek-powered chatbot!",
        "data": {}
    }), 200)

# Initialize index
if not os.path.exists("docs"):
    app.logger.warning("The 'docs' directory does not exist. Please create it and add your documents.")
    os.makedirs("docs", exist_ok=True)

construct_index("docs")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


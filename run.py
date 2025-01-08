from app import create_app
from flask import Flask, Response, request, jsonify
from typing import Generator
from custom_llm import CustomLLM
from langchain.chains.conversation.memory import ConversationBufferMemory

# Define configuration
BASE_URL = "http://10.124.68.81:10000"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"

# Initialize CustomLLM and Conversation Memory
custom_llm = CustomLLM(base_url=BASE_URL, model=MODEL_NAME, headers=HEADERS)
memory = ConversationBufferMemory(return_messages=True)

# Flask app initialization
app = create_app()

# Instruction template with conversation history
instruction_template = """
You are a helpful AI assistant. You will answer questions only in Vietnamese if there are no additional request.

Chat conversation:
{history}

Question:
{question}
"""

def get_prompt(question: str) -> str:
    """
    Create the prompt for the LLM based on the question and conversation history.

    :param question: The user's current question.
    :return: The formatted prompt string with conversation history.
    """
    history = memory.load_memory_variables({}).get('history', "")
    return instruction_template.format(history=history, question=question)

@app.route('/api/chat', methods=['POST'])
def stream_response():
    """
    Endpoint to stream response from the LLM as incremental tokens, with conversation memory.
    """
    data = request.json
    question = data.get("message", "")

    if not question:
        return jsonify({"error": "Message is required"}), 400

    # Add the user's question to memory
    memory.save_context({"input": question}, {"output": ""})

    # Create the prompt with conversation history
    prompt = get_prompt(question)

    def generate() -> Generator[str, None, None]:
        try:
            response_text = ""
            for token in custom_llm._call(prompt):
                response_text += token
                yield token  # Stream each token back to the client

            # Save the AI's response in memory
            memory.save_context({"input": question}, {"output": response_text})

        except Exception as e:
            yield f"[ERROR] {str(e)}\n\n"

    return Response(generate(), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)

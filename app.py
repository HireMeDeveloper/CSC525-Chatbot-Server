# Imports and dependencies
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

# Clean model output into JSON before returning it to the frontend
def extractJsonPayload(rawText):
    text = (rawText or "").strip()
    if not text:
        raise ValueError("Empty model response")
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if isinstance(payload, dict) and "game_idea" in payload:
        return payload
    if isinstance(payload, dict) and "gameIdea" in payload:
        return {"message": payload.get("message", ""), "game_idea": payload["gameIdea"]}
    raise ValueError("Model response was not valid JSON")

# Load the local model files from the project folder
def loadModelAndTokenizer():
    localModelPath = Path(__file__).resolve().parent / "model"
    if not localModelPath.exists():
        raise FileNotFoundError(f"Local model directory not found: {localModelPath}")

    tokenizerInstance = AutoTokenizer.from_pretrained(str(localModelPath), local_files_only=True)
    modelInstance = AutoModelForCausalLM.from_pretrained(
        str(localModelPath),
        device_map="auto",
        local_files_only=True,
    )
    return tokenizerInstance, modelInstance

# Create the Flask app and load the model
app = Flask(__name__)
CORS(app)
try:
    tokenizer, model = loadModelAndTokenizer()
except Exception:
    tokenizer = None
    model = None

# Load the rules file
rulesFile = Path(__file__).parent / "rules.md"
systemPrompt = ""
if rulesFile.exists():
    with open(rulesFile, encoding="utf-8") as file:
        systemPrompt = file.read().strip()

# Endpoint to handle chat requests
chatSessions = {}


@app.route("/api/chat", methods=["POST"])
def chat():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 503

    payload = request.get_json(silent=True) or {}
    sessionId = str(payload.get("session_id") or "default-session").strip() or "default-session"
    userMessage = str(payload.get("inputs", "")).strip()
    if not userMessage:
        return jsonify({"error": "No message"}), 400

    history = chatSessions.setdefault(sessionId, [])
    history.append({"role": "user", "content": userMessage})

    messages = [{"role": "system", "content": systemPrompt}]
    messages.extend(history)
    chatText = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokenizerInputs = tokenizer(chatText, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **tokenizerInputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    generatedText = tokenizer.decode(
        outputs[0][tokenizerInputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    responsePayload = extractJsonPayload(generatedText)
    history.append({"role": "assistant", "content": json.dumps(responsePayload)})
    return jsonify(responsePayload)

# Endpoint to reset the chat session for a given session ID
@app.route("/api/chat/reset", methods=["POST"])
def resetChat():
    payload = request.get_json(silent=True) or {}
    sessionId = str(payload.get("session_id") or "default-session").strip() or "default-session"
    chatSessions.pop(sessionId, None)
    return jsonify({"ok": True, "session_id": sessionId})

# App entry
if __name__ == "__main__":
    portNumber = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=portNumber, debug=False)

import os
import json
from flask import Flask, render_template, request, jsonify

# Import helpers from src package
from src.model_utils import (
    load_model,
    load_classification_model,
    load_categories,
    build_input_row,
    build_classification_input_row,
    predict_rate,
    predict_classification_category,
)

app = Flask(__name__)

# --- Load ML Models & Categories on Startup ---
regression_model = load_model()
classification_model = load_classification_model()
categories = load_categories()

# --- Lazy Initialized RAG Chat Engine ---
# Initialized on demand to prevent Gunicorn memory spike during boot
chat_engine = None

def get_chat_engine():
    global chat_engine
    if chat_engine is None:
        try:
            print("Initializing RAG Chat Engine lazily...")
            from src.RAG_V3 import build_index, SYSTEM_PROMPT
            rag_index = build_index()
            if rag_index:
                chat_engine = rag_index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    similarity_top_k=3,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.4,
                )
        except Exception as e:
            print(f"Warning: RAG Chat Engine lazy initialization failed: {e}")
            chat_engine = None
    return chat_engine


# --- Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_values = {
        "online_order": "Yes",
        "book_table": "Yes",
        "votes": 250,
        "approx_cost": 600,
        "rest_type": categories.get("rest_type", ["Casual Dining"])[0],
        "listed_in_type": categories.get("listed_in(type)", ["Delivery"])[0],
        "cuisines": categories.get("cuisines", ["North Indian"])[0],
        "dish_liked": categories.get("dish_liked", ["Pasta"])[0],
    }

    if request.method == "POST":
        form_values["online_order"] = request.form.get("online_order", "Yes")
        form_values["book_table"] = request.form.get("book_table", "Yes")
        form_values["votes"] = request.form.get("votes", 0)
        form_values["approx_cost"] = request.form.get("approx_cost", 0)
        form_values["rest_type"] = request.form.get("rest_type", "")
        form_values["listed_in_type"] = request.form.get("listed_in_type", "")
        form_values["cuisines"] = request.form.get("cuisines", "")
        form_values["dish_liked"] = request.form.get("dish_liked", "")

        if regression_model is not None:
            input_df = build_input_row(
                online_order=form_values["online_order"],
                book_table=form_values["book_table"],
                votes=form_values["votes"],
                rest_type=form_values["rest_type"],
                dish_liked=form_values["dish_liked"],
                cuisines=form_values["cuisines"],
                approx_cost=form_values["approx_cost"],
                listed_in_type=form_values["listed_in_type"],
            )
            raw_pred = predict_rate(regression_model, input_df)
            prediction = round(raw_pred, 2)

    return render_template(
        "index.html",
        form_values=form_values,
        categories=categories,
        prediction=prediction,
        model_missing=(regression_model is None),
    )


@app.route("/api/predict/classification", methods=["POST"])
def handle_classification():
    if classification_model is None:
        return jsonify({
            "status": "error",
            "message": "Classification model artifact missing. Check artifacts/dt_classifier_without_imputation.pkl"
        }), 500

    data = request.get_json() or {}

    try:
        input_row = build_classification_input_row(
            online_order=data.get("online_order", "Yes"),
            book_table=data.get("book_table", "Yes"),
            votes=data.get("votes", 0),
            location=data.get("location", "BTM"),
            approx_cost=data.get("approx_cost", 0),
            categories=categories,
        )

        predicted_label = predict_classification_category(classification_model, input_row)

        return jsonify({
            "status": "success",
            "predicted_category": predicted_label
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/rag/chat", methods=["POST"])
def handle_rag_chat():
    engine = get_chat_engine()
    if engine is None:
        return jsonify({
            "status": "error",
            "message": "RAG Chatbot is currently offline or GROQ_API_KEY is not set."
        }), 500

    data = request.get_json() or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"status": "error", "message": "Empty query."}), 400

    try:
        response = engine.chat(user_message)
        return jsonify({
            "status": "success",
            "response": str(response)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

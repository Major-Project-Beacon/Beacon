import os
import json
import re
import warnings

# Suppress Hugging Face symlink warnings on Windows
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import TypedDict
from dotenv import load_dotenv
from pymongo import MongoClient
from google import genai
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION & CONNECTIONS
# ============================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATABASE_NAME = "beacon_memory"
COLLECTION_NAME = "meeting_memories"
VECTOR_INDEX_NAME = "vector_index"
SIMILARITY_THRESHOLD = 0.65
VECTOR_SEARCH_CANDIDATES = 5

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer("alisha2006/beacon-minilm-v1")


# ============================================================
# STATE & NORMALIZATION
# ============================================================

class MemoryState(TypedDict, total=False):
    meeting_summary: str
    facts: list
    new_memories: list
    relevant_memories: list
    comparisons: list
    alerts: list
    change_detected: bool


def normalize(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_value(text):
    if text is None:
        return ""
    text = re.sub(r"[^\w\s]", " ", str(text).strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    normalized_words = []
    for word in words:
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        normalized_words.append(word)
    return " ".join(normalized_words)


# ============================================================
# NODES (SILENT EXECUTION)
# ============================================================

def extract_facts(state: MemoryState):
    meeting_summary = state["meeting_summary"]
    prompt = f"""
You are a general-purpose long-term meeting memory extractor.
Extract important factual information from the meeting summary that may be useful in future meetings.

Return ONLY a valid JSON array of objects with keys: "type", "topic", "value", "source_text".
If something changed or was updated, extract ONLY the NEW value.

Meeting summary:
{meeting_summary}
"""
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    response_text = response.text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)
    
    facts = json.loads(response_text)
    return {"facts": facts, "new_memories": []}


def create_embeddings(state: MemoryState):
    new_memories = []
    for fact in state["facts"]:
        text = f"Type: {fact.get('type', '')}. Topic: {fact.get('topic', '')}. Value: {fact.get('value', '')}. Source: {fact.get('source_text', '')}"
        embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()
        new_memories.append({
            "type": fact.get("type", ""),
            "topic": fact.get("topic", ""),
            "value": fact.get("value", ""),
            "source_text": fact.get("source_text", ""),
            "text": text,
            "embedding": embedding
        })
    return {"new_memories": new_memories}


def semantic_search(state: MemoryState):
    all_relevant_memories = []
    for current_memory in state["new_memories"]:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": current_memory["embedding"],
                    "numCandidates": VECTOR_SEARCH_CANDIDATES,
                    "limit": VECTOR_SEARCH_CANDIDATES
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "type": 1,
                    "topic": 1,
                    "value": 1,
                    "source_text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ])
        for memory in list(results):
            if float(memory.get("score", 0)) >= SIMILARITY_THRESHOLD:
                all_relevant_memories.append({
                    "current_memory": current_memory,
                    "previous_memory": memory
                })
    return {"relevant_memories": all_relevant_memories}


def same_fact_check(state: MemoryState):
    validated_memories = []
    for item in state.get("relevant_memories", []):
        current = item["current_memory"]
        previous = item["previous_memory"]
        prompt = f"""
Determine whether these two meeting memories refer to the SAME UNDERLYING FACT, decision, or property.
Do NOT decide if the values are equal. Return ONLY: YES or NO.

PREVIOUS: Topic: {previous.get('topic', '')} | Value: {previous.get('value', '')} | Source: {previous.get('source_text', '')}
CURRENT: Topic: {current.get('topic', '')} | Value: {current.get('value', '')} | Source: {current.get('source_text', '')}
"""
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        if "yes" in normalize_value(response.text):
            validated_memories.append(item)
    return {"relevant_memories": validated_memories}


def compare_memories(state: MemoryState):
    comparisons = []
    change_detected = False
    for item in state.get("relevant_memories", []):
        current = item["current_memory"]
        previous = item["previous_memory"]
        if normalize_value(current.get("value")) != normalize_value(previous.get("value")):
            change_detected = True
            comparisons.append({
                "topic": current.get("topic") or previous.get("topic"),
                "previous_value": previous.get("value"),
                "current_value": current.get("value")
            })
    return {"comparisons": comparisons, "change_detected": change_detected}


def generate_alert(state: MemoryState):
    alerts = []
    for comp in state.get("comparisons", []):
        alerts.append(f"{comp['topic']}: {comp['previous_value']} → {comp['current_value']}")
    return {"alerts": alerts}


def store_memories(state: MemoryState):
    for memory in state.get("new_memories", []):
        existing = collection.find_one({
            "type": {"$regex": "^" + re.escape(memory.get("type", "")) + "$", "$options": "i"},
            "topic": {"$regex": "^" + re.escape(memory.get("topic", "")) + "$", "$options": "i"},
            "value": {"$regex": "^" + re.escape(memory.get("value", "")) + "$", "$options": "i"}
        })
        if not existing:
            collection.insert_one(memory)
    return {}


def check_change(state: MemoryState):
    return "generate_alert" if state.get("change_detected", False) else "store_memories"


# ============================================================
# GRAPH DEFINITION
# ============================================================

workflow = StateGraph(MemoryState)

workflow.add_node("extract_facts", extract_facts)
workflow.add_node("create_embeddings", create_embeddings)
workflow.add_node("semantic_search", semantic_search)
workflow.add_node("same_fact_check", same_fact_check)
workflow.add_node("compare_memories", compare_memories)
workflow.add_node("generate_alert", generate_alert)
workflow.add_node("store_memories", store_memories)

workflow.add_edge(START, "extract_facts")
workflow.add_edge("extract_facts", "create_embeddings")
workflow.add_edge("create_embeddings", "semantic_search")
workflow.add_edge("semantic_search", "same_fact_check")
workflow.add_edge("same_fact_check", "compare_memories")
workflow.add_conditional_edges("compare_memories", check_change, {
    "generate_alert": "generate_alert",
    "store_memories": "store_memories"
})
workflow.add_edge("generate_alert", "store_memories")
workflow.add_edge("store_memories", END)

app = workflow.compile()


# ============================================================
# RUN & CLEAN OUTPUT
# ============================================================

if __name__ == "__main__":
    meeting_summary = input("\nEnter meeting summary:\n> ")
    result = app.invoke({"meeting_summary": meeting_summary})

    print("\n----------------------------------------")
    if result.get("change_detected"):
        print("⚠️ Change detected:")
        for alert in result.get("alerts", []):
            print(f"  • {alert}")
    else:
        print("✅ No change detected.")
    print("----------------------------------------\n")
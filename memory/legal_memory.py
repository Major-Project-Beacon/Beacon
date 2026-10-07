import os
import json
import re
import warnings

# Suppress warnings for a clean terminal
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
COLLECTION_NAME = "contract_clauses"
VECTOR_INDEX_NAME = "vector_index"

SIMILARITY_THRESHOLD = 0.55
VECTOR_SEARCH_CANDIDATES = 5

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer("alisha2006/beacon-minilm-v1")


# ============================================================
# STATE & NORMALIZATION
# ============================================================

class LegalState(TypedDict, total=False):
    contract_text: str
    facts: list
    new_memories: list
    relevant_memories: list
    comparisons: list
    alerts: list
    change_detected: bool


def normalize_legal_value(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ============================================================
# NODES
# ============================================================

def extract_legal_clauses(state: LegalState):
    contract_text = state["contract_text"]
    prompt = f"""
You are a specialized legal clause and obligation extractor.
Extract all binding legal terms, obligations, and restrictions from the provided contract text.

Focus on:
- Financial Terms (payment timelines, fees, penalties, interest)
- Obligations & Responsibilities (who is bound to do what)
- Termination & Duration (notice periods, renewal, end dates)
- Liability, Indemnification & Caps
- Governing Law & Dispute Jurisdiction
- Confidentiality & IP Rights

For each clause, produce an object with:
- "type": category (e.g., "Payment Terms", "Liability", "Termination", "Jurisdiction")
- "topic": specific underlying obligation (e.g., "payment notice period", "governing state law", "liability cap")
- "value": the exact, specific condition, figure, or obligation
- "source_text": verbatim sentence or phrase from the text

Return ONLY a valid JSON array of objects.

Contract Text:
{contract_text}
"""
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={"automatic_function_calling": {"disable": True}}
    )
    response_text = response.text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    facts = json.loads(response_text)
    
    print(f"\n[Debug] Node 1 - Extracted {len(facts)} legal clauses from text.")
    if len(facts) > 0:
        print(f"[Debug] Node 1 - First extracted topic: {facts[0].get('topic')}")
        
    return {"facts": facts, "new_memories": []}


def create_embeddings(state: LegalState):
    new_memories = []
    for fact in state["facts"]:
        text = f"Category: {fact.get('type', '')}. Obligation: {fact.get('topic', '')}. Term: {fact.get('value', '')}. Source: {fact.get('source_text', '')}"
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


def semantic_search(state: LegalState):
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
        
        candidates = list(results)
        print(f"[Debug] Node 3 - Atlas returned {len(candidates)} raw vectors from DB.")
        
        for memory in candidates:
            score = float(memory.get("score", 0))
            if score >= SIMILARITY_THRESHOLD:
                all_relevant_memories.append({
                    "current_memory": current_memory,
                    "previous_memory": memory
                })
    
    print(f"[Debug] Node 3 - Candidates that passed the {SIMILARITY_THRESHOLD} threshold: {len(all_relevant_memories)}")
    return {"relevant_memories": all_relevant_memories}


def same_clause_check(state: LegalState):
    validated_memories = []
    for item in state.get("relevant_memories", []):
        current = item["current_memory"]
        previous = item["previous_memory"]
        prompt = f"""
You are comparing two legal contract clauses.
Determine whether they govern the SAME underlying legal topic.

CRITICAL RULES:
- Treat "Governing Law", "Jurisdiction", and "Venue" as the EXACT SAME TOPIC.
- Even if the values, states, or conditions are completely different, answer YES if they cover the same subject area.

PREVIOUS: Topic: {previous.get('topic', '')} | Value: {previous.get('value', '')}
CURRENT: Topic: {current.get('topic', '')} | Value: {current.get('value', '')}

Return ONLY: YES or NO.
"""
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        response_text = response.text.strip().lower()
        print(f"[Debug] Node 4 - Same-clause check for '{current.get('topic')}': {response_text}")
        
        if "yes" in response_text:
            validated_memories.append(item)
            
    return {"relevant_memories": validated_memories}


def compare_clauses(state: LegalState):
    comparisons = []
    change_detected = False
    for item in state.get("relevant_memories", []):
        current = item["current_memory"]
        previous = item["previous_memory"]
        
        prompt = f"""
Compare the previous legal condition with the new amended condition.
Has the actual legal rule, liability, value, or jurisdiction CHANGED?

IMPORTANT: If the new clause explicitly states it is "retaining", "keeping", or "maintaining" the previous condition, there is NO change.

PREVIOUS VALUE: {previous.get('value', '')}
NEW VALUE: {current.get('value', '')}

Return ONLY: CHANGED or UNCHANGED.
"""
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={"automatic_function_calling": {"disable": True}}
        )
        
        if "changed" in response.text.lower() and "unchanged" not in response.text.lower():
            change_detected = True
            comparisons.append({
                "topic": current.get("topic") or previous.get("topic"),
                "previous_value": previous.get("value"),
                "current_value": current.get("value")
            })
            
    return {"comparisons": comparisons, "change_detected": change_detected}


def generate_alert(state: LegalState):
    alerts = []
    for comp in state.get("comparisons", []):
        alerts.append(f"{comp['topic']}: {comp['previous_value']} → {comp['current_value']}")
    return {"alerts": alerts}


def store_memories(state: LegalState):
    stored_count = 0
    for memory in state.get("new_memories", []):
        existing = collection.find_one({
            "type": {"$regex": "^" + re.escape(memory.get("type", "")) + "$", "$options": "i"},
            "topic": {"$regex": "^" + re.escape(memory.get("topic", "")) + "$", "$options": "i"},
            "value": {"$regex": "^" + re.escape(memory.get("value", "")) + "$", "$options": "i"}
        })
        if not existing:
            collection.insert_one(memory)
            stored_count += 1
            
    print(f"[Debug] Node 7 - Saved {stored_count} new clauses to MongoDB.")
    return {}


def check_change(state: LegalState):
    return "generate_alert" if state.get("change_detected", False) else "store_memories"


# ============================================================
# GRAPH DEFINITION
# ============================================================

workflow = StateGraph(LegalState)

workflow.add_node("extract_legal_clauses", extract_legal_clauses)
workflow.add_node("create_embeddings", create_embeddings)
workflow.add_node("semantic_search", semantic_search)
workflow.add_node("same_clause_check", same_clause_check)
workflow.add_node("compare_clauses", compare_clauses)
workflow.add_node("generate_alert", generate_alert)
workflow.add_node("store_memories", store_memories)

workflow.add_edge(START, "extract_legal_clauses")
workflow.add_edge("extract_legal_clauses", "create_embeddings")
workflow.add_edge("create_embeddings", "semantic_search")
workflow.add_edge("semantic_search", "same_clause_check")
workflow.add_edge("same_clause_check", "compare_clauses")
workflow.add_conditional_edges("compare_clauses", check_change, {
    "generate_alert": "generate_alert",
    "store_memories": "store_memories"
})
workflow.add_edge("generate_alert", "store_memories")
workflow.add_edge("store_memories", END)

legal_app = workflow.compile()


# ============================================================
# RUN & OUTPUT
# ============================================================

if __name__ == "__main__":
    contract_text = input("\nEnter contract text or clause:\n> ")
    result = legal_app.invoke({"contract_text": contract_text})

    print("\n----------------------------------------")
    if result.get("change_detected"):
        print("⚠️ Contract discrepancy / revision detected:")
        for alert in result.get("alerts", []):
            print(f"  • {alert}")
    else:
        print("✅ No discrepancy detected.")
    print("----------------------------------------\n")
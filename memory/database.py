import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

client = MongoClient(os.environ["MONGODB_URI"])
db = client["beacon_memory_v2"]

# Two separate collections
summaries_collection = db["meeting_summaries"]
key_points_collection = db["meeting_key_points"]

# Ensure unique indexing on meeting_id
summaries_collection.create_index([("meeting_id", ASCENDING)], unique=True)
key_points_collection.create_index([("meeting_id", ASCENDING)], unique=True)


def store_meeting_data(summary_json, text_summary, meeting_id=None):
    """
    Stores summary data and key points/extracted facts into two separate 
    MongoDB collections, linked by the same meeting_id.
    """
    metadata = summary_json.get("meeting_metadata", {})
    meeting_id = (
        meeting_id
        or summary_json.get("meeting_id")
        or str(uuid.uuid4())
    )
    timestamp = datetime.now(timezone.utc)

    # 1. Collection 1: Store Meeting Summary & Metadata
    summary_document = {
        "meeting_id": meeting_id,
        "title": metadata.get("title", "Untitled"),
        "date": metadata.get("date", "Unknown"),
        "language": metadata.get("language", "en"),
        "text_summary": text_summary,
        "created_at": timestamp,
    }
    summaries_collection.update_one(
        {"meeting_id": meeting_id},
        {"$set": summary_document},
        upsert=True
    )

    # 2. Collection 2: Store Structured Key Points & Entities (for Neo4j / Analysis)
    key_points_document = {
        "meeting_id": meeting_id,
        "title": metadata.get("title", "Untitled"),
        "key_points": summary_json.get("key_points", []),
        "decisions": summary_json.get("decisions", []),
        "action_items": summary_json.get("action_items", []),
        "commitments": summary_json.get("commitments", []),
        "problems_and_risks": summary_json.get("problems_and_risks", []),
        "created_at": timestamp,
    }
    key_points_collection.update_one(
        {"meeting_id": meeting_id},
        {"$set": key_points_document},
        upsert=True
    )

    print(f"[MongoDB] Stored separated data for meeting_id: {meeting_id}")
    return meeting_id


def get_linked_meeting_data(meeting_id):
    """
    Retrieves and combines the summary and key points data using the same meeting_id.
    """
    summary_doc = summaries_collection.find_one({"meeting_id": meeting_id}, {"_id": 0})
    key_points_doc = key_points_collection.find_one({"meeting_id": meeting_id}, {"_id": 0})

    if not summary_doc and not key_points_doc:
        return None

    return {
        "meeting_id": meeting_id,
        "summary_info": summary_doc,
        "extracted_facts": key_points_doc
    }
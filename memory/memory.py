import json
from pathlib import Path
from database import store_meeting_data, get_linked_meeting_data
from graph_builder import create_graph_from_key_points, close_graph


def process_meeting_from_files(summary_path, meeting_path):
    # 1. Load summary JSON
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    
    # 2. Load structured meeting JSON (key points / facts)
    with open(meeting_path, "r", encoding="utf-8") as f:
        meeting_json = json.load(f)

    # Extract text summary and shared meeting_id
    meeting_id = summary_data.get("meeting_id") or meeting_json.get("meeting_id")
    text_summary = summary_data.get("text_summary", "")

    # 3. Store in MongoDB (split collections linked by meeting_id)
    store_meeting_data(meeting_json, text_summary, meeting_id=meeting_id)

    # 4. Build Neo4j Graph using key points
    try:
        create_graph_from_key_points(meeting_json, meeting_id)
        print(f"Graph relationships built successfully for meeting: {meeting_id}")
    except Exception as e:
        print(f"Graph creation failed: {e}. MongoDB records are safe.")
        raise

    return meeting_id


if __name__ == "__main__":
    try:
        summary_file = Path("sample_data/summary.json")
        meeting_file = Path("sample_data/meeting.json")

        if summary_file.exists() and meeting_file.exists():
            m_id = process_meeting_from_files(summary_file, meeting_file)
            print(f"\nSuccessfully processed and stored meeting: {m_id}")

            # Test Retrieval from MongoDB collections
            print("\n--- Testing Linked Retrieval ---")
            linked_data = get_linked_meeting_data(m_id)
            print(f"Title: {linked_data['summary_info']['title']}")
            print(f"Text Summary: {linked_data['summary_info']['text_summary']}")
            print(f"Decisions Mapped in Graph: {len(linked_data['extracted_facts']['decisions'])}")
        else:
            print("Sample data files not found in sample_data/ directory.")

    finally:
        close_graph()
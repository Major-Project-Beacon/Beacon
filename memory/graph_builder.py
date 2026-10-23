import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(
        os.environ["NEO4J_USERNAME"],
        os.environ["NEO4J_PASSWORD"]
    )
)


def create_graph_from_key_points(extracted_facts, meeting_id):
    """
    Takes the structured key points JSON and builds the Neo4j graph relationships.
    """
    # Fix: Safely extract title from meeting_metadata
    metadata = extracted_facts.get("meeting_metadata", {})
    title = metadata.get("title", extracted_facts.get("title", "Untitled Meeting"))
    
    with driver.session() as session:
        # 1. Create Meeting Node (Added name property for Neo4j display)
        session.run("""
            MERGE (m:Meeting {meeting_id: $meeting_id})
            SET m.title = $title, m.name = $title
        """, meeting_id=meeting_id, title=title)

        # 2. Process Decisions
        for i, item in enumerate(extracted_facts.get("decisions", [])):
            decision_id = f"{meeting_id}:decision:{i}"
            speaker = item.get("speaker")
            decision_text = item.get("decision", "")
            
            session.run("""
                MATCH (m:Meeting {meeting_id: $meeting_id})
                MERGE (d:Decision {id: $decision_id})
                SET d.text = $text, d.name = $text, d.status = $status
                MERGE (m)-[:HAS_DECISION]->(d)
            """, meeting_id=meeting_id, decision_id=decision_id, text=decision_text, status=item.get("status"))

            if speaker:
                session.run("""
                    MATCH (m:Meeting {meeting_id: $meeting_id}), (d:Decision {id: $decision_id})
                    MERGE (p:Participant {name: $speaker})
                    MERGE (p)-[:MADE_DECISION]->(d)
                    MERGE (p)-[:ATTENDED]->(m)
                """, meeting_id=meeting_id, decision_id=decision_id, speaker=speaker)

        # 3. Process Action Items
        for i, item in enumerate(extracted_facts.get("action_items", [])):
            action_id = f"{meeting_id}:action:{i}"
            owner = item.get("owner")
            task_text = item.get("task", "")
            
            session.run("""
                MATCH (m:Meeting {meeting_id: $meeting_id})
                MERGE (a:ActionItem {id: $action_id})
                SET a.task = $task, a.name = $task, a.deadline = $deadline, a.status = $status
                MERGE (m)-[:HAS_ACTION_ITEM]->(a)
            """, meeting_id=meeting_id, action_id=action_id, task=task_text, deadline=item.get("deadline"), status=item.get("status"))

            if owner:
                session.run("""
                    MATCH (m:Meeting {meeting_id: $meeting_id}), (a:ActionItem {id: $action_id})
                    MERGE (p:Participant {name: $owner})
                    MERGE (p)-[:ASSIGNED_TO]->(a)
                    MERGE (p)-[:ATTENDED]->(m)
                """, meeting_id=meeting_id, action_id=action_id, owner=owner)

        # 4. Process Risks
        for i, item in enumerate(extracted_facts.get("problems_and_risks", [])):
            risk_id = f"{meeting_id}:risk:{i}"
            speaker = item.get("speaker")
            issue_text = item.get("issue", "")
            
            session.run("""
                MATCH (m:Meeting {meeting_id: $meeting_id})
                MERGE (r:Risk {id: $risk_id})
                SET r.issue = $issue, r.name = $issue
                MERGE (m)-[:HAS_RISK]->(r)
            """, meeting_id=meeting_id, risk_id=risk_id, issue=issue_text)

            if speaker:
                session.run("""
                    MATCH (m:Meeting {meeting_id: $meeting_id}), (r:Risk {id: $risk_id})
                    MERGE (p:Participant {name: $speaker})
                    MERGE (p)-[:RAISED_RISK]->(r)
                    MERGE (p)-[:ATTENDED]->(m)
                """, meeting_id=meeting_id, risk_id=risk_id, speaker=speaker)

    print(f"[Neo4j] Graph successfully built for meeting_id: {meeting_id}")


def close_graph():
    driver.close()
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cogbot import config
from cogbot.memory.canonical_store import CanonicalMemoryStore
from cogbot.modules.ltm_graph import MemoryGraph


def backfill_legacy_insights(graph_path: str, embeddings_path: str, db_path: str) -> int:
    graph = MemoryGraph(graph_path=graph_path, embeddings_path=embeddings_path)
    store = CanonicalMemoryStore(db_path=db_path)
    migrated = 0

    for insight in list(graph.insights.values()):
        if not insight.summary:
            continue

        related_entity_ids = []
        evidence_episode_ids = []
        for edge_id in insight.edges.keys():
            node = graph.get_node(edge_id)
            if not node:
                continue
            if getattr(node, "type", "") == "entity":
                related_entity_ids.append(getattr(node, "user_id", ""))
            elif getattr(node, "type", "") == "episode":
                evidence_episode_ids.append(node.node_id)

        graph.add_or_update_note(
            summary=insight.summary,
            note_type="theme",
            tags=["legacy_insight"],
            confidence=insight.confidence,
            related_entity_ids=[entity_id for entity_id in related_entity_ids if entity_id],
            evidence_episode_ids=evidence_episode_ids,
            embedding=graph._embeddings_cache.get(insight.node_id),
        )
        migrated += 1

    graph.save_all()
    return migrated


def main():
    parser = argparse.ArgumentParser(description="Backfill legacy Insight nodes into Note nodes.")
    parser.add_argument("--graph-path", default=config.LTM_GRAPH_PATH)
    parser.add_argument("--embeddings-path", default=config.LTM_EMBEDDINGS_PATH)
    parser.add_argument("--db-path", default=config.CANONICAL_MEMORY_DB_PATH)
    args = parser.parse_args()

    migrated = backfill_legacy_insights(
        graph_path=args.graph_path,
        embeddings_path=args.embeddings_path,
        db_path=args.db_path,
    )
    print(f"Backfilled {migrated} legacy insights into notes.")


if __name__ == "__main__":
    main()

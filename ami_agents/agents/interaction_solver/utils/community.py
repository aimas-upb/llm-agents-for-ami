"""Community model and utilities for multi-agent planning."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Community:
    """Represents a community of agents for collaborative planning."""

    community_id: str
    member_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    match_threshold: float = 0.5

    def matches_intents(self, intents: List[str]) -> float:
        """
        Calculate similarity score between provided intents and community keywords.
        
        Returns:
            float: Similarity score 0.0-1.0. If no keywords, returns 1.0 (accept all).
        """
        if not self.keywords:
            return 1.0

        if not intents:
            return 0.0

        # Simple keyword matching: count how many intent words match keywords
        intent_text = " ".join(intents).lower()
        matched_keywords = sum(1 for kw in self.keywords if kw.lower() in intent_text)

        return matched_keywords / len(self.keywords) if self.keywords else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "community_id": self.community_id,
            "member_ids": self.member_ids,
            "keywords": self.keywords,
            "match_threshold": self.match_threshold,
        }


def build_communities(raw: Any) -> List[Community]:
    """
    Build Community list from YAML configuration.
    
    Args:
        raw: Raw config from YAML (list of dicts or list of Community-like objects)
    
    Returns:
        List of Community instances.
    """
    if not isinstance(raw, list):
        return []

    communities: List[Community] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                community = Community(
                    community_id=str(item.get("community_id", "")),
                    member_ids=item.get("member_ids", []),
                    keywords=item.get("keywords", []),
                    match_threshold=float(item.get("match_threshold", 0.5)),
                )
                if community.community_id:
                    communities.append(community)
            except Exception:
                pass

    return communities

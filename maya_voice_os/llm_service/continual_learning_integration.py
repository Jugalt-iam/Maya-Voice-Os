"""
Continual Learning Integration — adapted from the original.

DIFFERENCE FROM THE ORIGINAL: the original tracked feedback per "expert"
from a Mixture-of-Experts system (multiple separately fine-tuned models we
don't run — see the file-by-file review for why MoE was rejected). This
version tracks feedback per ROUTE instead — "fast_path", "response_selector",
or "llm:<provider>" (e.g. "llm:groq", "llm:cerebras") — which is the
equivalent concept for this repo's architecture and is genuinely more
useful here: it tells you which free LLM provider is actually giving good
answers, not just an abstract "expert" score.

Also different: `_apply_adaptations` in the original silently mutated an
`expert_manager.model_configs` object we don't have (temperature, max
tokens per expert). We don't own model weights or per-call sampling
parameters for free-tier hosted providers, so there's nothing to silently
mutate. This version logs concrete, actionable recommendations instead
(e.g. "route llm:groq is underperforming — consider reordering
LLM_PROVIDER_ORDER or revising the system prompt") rather than pretending
to apply a change that wouldn't actually do anything.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("continual-learning")


class ContinualLearningManager:
    """Tracks feedback per route and surfaces actionable recommendations
    when a route's recent performance drops — does not (and cannot, for
    free hosted providers) silently reconfigure anything on its own."""

    def __init__(self):
        self.feedback_storage: Dict[str, List[dict]] = {}
        self.learning_metrics: Dict[str, dict] = {}
        self.adaptation_threshold = 0.7  # rating (0-1 normalized) below this triggers a recommendation
        self.min_feedback_samples = 10

    def record_feedback(self, conversation_id: str, feedback_data: Dict[str, Any]) -> None:
        """
        feedback_data expects:
            rating: 1-5 caller/operator satisfaction rating
            route_used: "fast_path" | "response_selector" | "llm:<provider>"
            quality_rating, relevance_rating, satisfaction: optional 1-5 sub-scores
            suggestions: optional free-text improvement note
            response_text: what was actually said (truncated for storage)
        """
        route = feedback_data.get("route_used", "unknown")
        entry = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "rating": feedback_data.get("rating", 0),
            "quality_rating": feedback_data.get("quality_rating", 0),
            "relevance_rating": feedback_data.get("relevance_rating", 0),
            "satisfaction": feedback_data.get("satisfaction", 0),
            "suggestions": feedback_data.get("suggestions", ""),
            "response_text": (feedback_data.get("response_text") or "")[:500],
        }

        self.feedback_storage.setdefault(route, []).append(entry)
        self._update_metrics(route, entry)
        self._check_for_recommendation(route)
        logger.info(f"Recorded feedback for route '{route}': rating={entry['rating']}")

    def _update_metrics(self, route: str, entry: dict) -> None:
        metrics = self.learning_metrics.setdefault(route, {
            "total_interactions": 0,
            "average_rating": 0.0,
            "recent_ratings": [],
            "issue_counts": {},
        })
        metrics["total_interactions"] += 1

        rating = entry["rating"]
        if metrics["total_interactions"] == 1:
            metrics["average_rating"] = rating
        else:
            alpha = 0.1  # exponential moving average
            metrics["average_rating"] = alpha * rating + (1 - alpha) * metrics["average_rating"]

        metrics["recent_ratings"].append(rating)
        if len(metrics["recent_ratings"]) > 20:
            metrics["recent_ratings"] = metrics["recent_ratings"][-20:]

        suggestions = (entry.get("suggestions") or "").lower()
        issue_map = {
            "slow": ("slow", "speed"),
            "irrelevant": ("irrelevant", "off-topic"),
            "tone": ("rude", "impolite"),
            "unclear": ("unclear", "confusing"),
        }
        for issue_key, markers in issue_map.items():
            if any(m in suggestions for m in markers):
                metrics["issue_counts"][issue_key] = metrics["issue_counts"].get(issue_key, 0) + 1

    def _check_for_recommendation(self, route: str) -> None:
        metrics = self.learning_metrics[route]
        if metrics["total_interactions"] < self.min_feedback_samples:
            return

        recent = metrics["recent_ratings"][-10:]
        if len(recent) < 10:
            return

        recent_avg_normalized = (sum(recent) / len(recent)) / 5.0  # ratings are 1-5
        if recent_avg_normalized < self.adaptation_threshold:
            self._log_recommendation(route, recent_avg_normalized, metrics["issue_counts"])

    def _log_recommendation(self, route: str, score: float, issue_counts: Dict[str, int]) -> None:
        top_issues = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        issue_summary = ", ".join(f"{k} ({v}x)" for k, v in top_issues[:3]) or "no specific pattern in feedback text"

        if route.startswith("llm:"):
            suggestion = (
                f"Consider reordering LLM_PROVIDER_ORDER to deprioritize this provider, "
                f"or revising the system prompt in identity/*.yaml — recent issues: {issue_summary}"
            )
        elif route == "response_selector":
            suggestion = (
                f"Consider editing the text bank in llm-service/smart_response_selector.py "
                f"for this intent — recent issues: {issue_summary}"
            )
        elif route == "fast_path":
            suggestion = (
                f"Consider editing the matching playbook's response text in playbooks/*.yaml "
                f"— recent issues: {issue_summary}"
            )
        else:
            suggestion = f"Recent issues: {issue_summary}"

        logger.warning(
            f"RECOMMENDATION: route '{route}' recent satisfaction score is "
            f"{score:.0%} (below {self.adaptation_threshold:.0%} threshold). {suggestion}"
        )

    def get_learning_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"routes": {}}
        total_interactions = 0
        total_rating_sum = 0.0

        for route, metrics in self.learning_metrics.items():
            stats["routes"][route] = {
                "interactions": metrics["total_interactions"],
                "average_rating": round(metrics["average_rating"], 2),
                "top_issues": sorted(metrics["issue_counts"].items(), key=lambda kv: kv[1], reverse=True)[:3],
            }
            total_interactions += metrics["total_interactions"]
            total_rating_sum += metrics["average_rating"] * metrics["total_interactions"]

        if total_interactions > 0:
            stats["overall_average_rating"] = round(total_rating_sum / total_interactions, 2)
            stats["overall_interactions"] = total_interactions

        return stats


_continual_learning_manager: Optional[ContinualLearningManager] = None


def get_continual_learning_manager() -> ContinualLearningManager:
    global _continual_learning_manager
    if _continual_learning_manager is None:
        _continual_learning_manager = ContinualLearningManager()
    return _continual_learning_manager

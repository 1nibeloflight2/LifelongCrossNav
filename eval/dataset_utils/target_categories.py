"""Strict HM3D semantic-name rules for the six ObjectNav targets.

The HM3D raw labels are fine grained (for example ``wall tv`` and
``toilet paper``), while the benchmark uses six canonical target classes.
Never use substring matching here: it both admits accessories such as
``bedside lamp`` and misses valid aliases such as ``wall tv``.
"""

import re
from typing import Dict, FrozenSet, Optional


TARGET_CATEGORIES: FrozenSet[str] = frozenset({
    "chair",
    "bed",
    "toilet",
    "plant",
    "sofa",
    "tv_monitor",
})


def normalize_semantic_name(name: str) -> str:
    """Normalize separators/case without weakening exact-name matching."""
    value = str(name or "").strip().lower()
    value = re.sub(r"[_/\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Exact aliases only.  The list covers all relevant labels currently present
# in HM3D-v0.2 val and a few unambiguous spelling variants.
TARGET_CATEGORY_ALIASES: Dict[str, FrozenSet[str]] = {
    "chair": frozenset({
        "chair", "armchair", "bar chair", "bean bag chair", "camping chair",
        "computer chair", "desk chair", "dining chair", "folded chair",
        "folding chair", "highchair", "high chair", "lounge chair",
        "office chair", "patio chair", "rocking chair", "sofa chair",
        "chair stand", "recliner",
    }),
    "bed": frozenset({
        "bed", "bed small", "bedframe", "bed frame", "bunk bed",
        "single bed", "double bed", "queen bed", "king bed", "crib",
    }),
    "toilet": frozenset({
        "toilet", "toilet seat",
    }),
    "plant": frozenset({
        "plant", "decorative plant", "flowerpot", "flower pot",
        "potted plant", "indoor plant", "houseplant", "planter",
        "flower vase", "flower stand", "dried flowers","flower",
    }),
    "sofa": frozenset({
        "sofa", "couch", "circular sofa", "l shaped sofa", "sectional sofa",
        "sofa seat", "sofa set", "loveseat", "settee",
    }),
    "tv_monitor": frozenset({
        "tv", "led tv", "wall tv", "television", "monitor",
        "computer monitor", "flat screen tv", "flatscreen tv",
        "wall mounted tv",
    }),
}


# Kept explicitly for audits and documentation. Exact aliases already reject
# these names; listing them makes the intended boundary reviewable.
TARGET_CATEGORY_EXCLUSIONS: Dict[str, FrozenSet[str]] = {
    "chair": frozenset(),
    "bed": frozenset({
        "bed light", "bed sheet", "bed stand", "bed table", "bedroom table",
        "bedside cabinet", "bedside lamp", "bedside table", "dog bed",
    }),
    "toilet": frozenset({
        "toilet brush", "toilet brush holder", "toilet cleaner",
        "toilet paper", "toilet paper dispenser", "toiletry",
        "wall toilet paper", "potty",
    }),
    "plant": frozenset({
         "pot",
    }),
    "sofa": frozenset(),
    "tv_monitor": frozenset({"tv remote", "tv stand"}),
}


def canonical_target_category(category: str) -> Optional[str]:
    """Return the benchmark key for a canonical category/key suffix."""
    raw = str(category or "").strip().lower()
    if raw in TARGET_CATEGORIES:
        return raw
    normalized = normalize_semantic_name(raw)
    if normalized == "tv monitor":
        return "tv_monitor"
    for candidate in TARGET_CATEGORIES:
        if raw.endswith("_" + candidate):
            return candidate
        if normalized.endswith(" " + normalize_semantic_name(candidate)):
            return candidate
    return None


def matches_target_category(category: str, semantic_name: str) -> bool:
    """Whether a raw HM3D label belongs to one of the six target classes."""
    canonical = canonical_target_category(category)
    if canonical is None:
        return False
    name = normalize_semantic_name(semantic_name)
    if name in TARGET_CATEGORY_EXCLUSIONS[canonical]:
        return False
    return name in TARGET_CATEGORY_ALIASES[canonical]

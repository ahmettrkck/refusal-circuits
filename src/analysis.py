"""Graph comparison and intervention helpers."""

from __future__ import annotations

import torch
from collections import Counter
from dataclasses import dataclass


@dataclass
class FeatureActivation:
    layer: int
    feature_idx: int
    activation: float
    label: str | None = None


def extract_top_features(graph, top_k=50, min_effect=0.01):
    """Pull out the top-k transcoder features by total effect from an attribution graph."""
    features = []
    for node in graph.nodes:
        if hasattr(node, "layer") and hasattr(node, "feature_idx"):
            effect = abs(node.total_effect) if hasattr(node, "total_effect") else 0.0
            if effect >= min_effect:
                features.append(FeatureActivation(
                    layer=node.layer,
                    feature_idx=node.feature_idx,
                    activation=effect,
                    label=getattr(node, "label", None),
                ))
    features.sort(key=lambda f: f.activation, reverse=True)
    return features[:top_k]


def compare_graphs(harmful_features, benign_features):
    """Diff two feature lists — returns harmful-only, benign-only, and shared sets."""
    harmful_set = {(f.layer, f.feature_idx) for f in harmful_features}
    benign_set = {(f.layer, f.feature_idx) for f in benign_features}

    harmful_only = harmful_set - benign_set
    benign_only = benign_set - harmful_set
    shared = harmful_set & benign_set

    harmful_lookup = {(f.layer, f.feature_idx): f for f in harmful_features}
    benign_lookup = {(f.layer, f.feature_idx): f for f in benign_features}

    return {
        "refusal_candidates": sorted(
            [harmful_lookup[k] for k in harmful_only],
            key=lambda f: f.activation, reverse=True,
        ),
        "benign_only": sorted(
            [benign_lookup[k] for k in benign_only],
            key=lambda f: f.activation, reverse=True,
        ),
        "shared": sorted(
            [harmful_lookup[k] for k in shared],
            key=lambda f: f.activation, reverse=True,
        ),
    }


def find_consistent_refusal_features(all_pair_results, min_frequency=3):
    """Find features showing up in harmful-only across multiple prompt pairs."""
    counter = Counter()
    for result in all_pair_results:
        for feat in result["refusal_candidates"]:
            counter[(feat.layer, feat.feature_idx)] += 1

    return [
        (layer, idx, count)
        for (layer, idx), count in counter.most_common()
        if count >= min_frequency
    ]


def run_intervention(model, prompt, feature_layer, feature_idx, scale=0.0, max_new_tokens=50):
    """Set a feature to a given scale and generate. scale=0 ablates, scale>1 amplifies."""
    interventions = {(feature_layer, feature_idx): scale}
    return model.generate_with_interventions(
        prompt, interventions=interventions, max_new_tokens=max_new_tokens,
    )


def ablation_experiment(model, prompt, candidate_features, max_new_tokens=50):
    """Ablate each candidate refusal feature one at a time, compare to baseline."""
    baseline = model.generate(prompt, max_new_tokens=max_new_tokens)
    results = {"baseline": baseline}

    for layer, feature_idx in candidate_features:
        results[f"ablate_L{layer}_F{feature_idx}"] = run_intervention(
            model, prompt, layer, feature_idx, scale=0.0, max_new_tokens=max_new_tokens,
        )

    return results

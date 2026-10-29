"""detector.classes can be ids, names, a mix, or empty (= every class)."""
from __future__ import annotations

import pytest

from src.detector import merge_names, resolve_classes

COCO = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CUSTOM = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "rickshaw"}


def test_ids_pass_through_sorted_and_deduplicated():
    assert resolve_classes([7, 2, 3, 5, 2], COCO) == [2, 3, 5, 7]
    assert resolve_classes(["2", 3], COCO) == [2, 3]                      # numeric strings from YAML / CLI


def test_names_are_resolved_case_insensitively_for_a_custom_model():
    assert resolve_classes(["Car", "rickshaw", " Truck "], CUSTOM) == [0, 3, 4]
    assert resolve_classes(["car", 4], CUSTOM) == [0, 4]                  # mixed


def test_empty_means_every_class():
    assert resolve_classes(None, COCO) is None
    assert resolve_classes([], COCO) is None


def test_config_class_names_never_rename_a_fine_tuned_models_classes():
    coco_overrides = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}      # the default config.yaml entries
    merged = merge_names(CUSTOM, coco_overrides)
    assert merged == CUSTOM                                                             # the model's names, untouched
    assert resolve_classes(["bus", "rickshaw"], merged) == [2, 4]                       # no stale ids 5 / 7 with duplicate names
    assert merge_names({}, {2: "car"}) == {2: "car"}                                    # fallback only for a model without names


def test_unknown_name_gives_a_helpful_error():
    with pytest.raises(ValueError, match="rickshaw.*Available"):
        resolve_classes(["car", "rickshaw"], COCO)                        # the stock model has no rickshaw class

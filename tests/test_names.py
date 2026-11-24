"""Standard bone names: the templates, and the structure match against real skeletons."""
import json

import pytest
import yaml

from conftest import REPO
from glbrig import GlbRig

import sys

sys.path.insert(0, str(REPO))
from engine import names  # noqa: E402


def test_templates_are_trees_in_depth_first_order():
    for layout in ("body22", "full52-yaml", "full52-dfs"):
        bare, parents = names.template(layout)
        assert parents[0] == -1 and all(0 <= p < i for i, p in enumerate(parents) if i)
        assert len(set(bare)) == len(bare)


@pytest.mark.parametrize("model", ["skintokens", "unirig"])
def test_names_are_the_models_own_mixamo_names(engine, model):
    config = yaml.safe_load((engine / "models" / model / "configs/skeleton/mixamo.yaml").read_text(encoding="utf-8"))
    body, hand = config["parts"]["body"], config["parts"]["hand"]
    assert [names.PREFIX + n for n in names.BODY] == body
    assert [names.PREFIX + n for n in names.HAND] == hand
    _, full = names.match(names.template("full52-yaml")[1])
    assert full == body + hand


def test_no_match_for_other_counts_or_structures():
    assert names.match([-1, 0, 1]) is None
    _, parents = names.template("body22")
    changed = list(parents)
    changed[5] = 0
    assert names.match(changed) is None
    assert names.decide(parents, None, asked=False)[0] is None


def _work_b_runs(final):
    rows = json.loads((final / "work-B/runs/layout.json").read_text(encoding="utf-8"))
    for row in rows:
        stem = row["model"].removesuffix(".glb")
        path = final / "work-B/runs" / stem / f"{row['class']}_seed_{row['seed']}" / "rigged.glb"
        if path.exists():
            yield row, path


def test_structure_match_agrees_with_the_docker_round(final):
    """work-B decided, from SkinTokens' tokens, which skeletons have the Mixamo layout. The same skeletons,
    read back from their rigged files, get the same answer here."""
    checked = matched = 0
    for row, path in _work_b_runs(final):
        rig = GlbRig(path)
        found = names.match(rig.parents)
        expected = bool(row.get("parents_match_mixamo"))
        assert (found is not None) == expected, (path, row)
        if found is not None:
            expected_layout = "body22" if row["bones"] == 22 else "full52-" + row["layout"]
            assert found[0] == expected_layout, (path, row)
            assert names.placement_warnings(rig.joints, found[1]) == [], path
            matched += 1
        checked += 1
    assert checked >= 12 and matched >= 4, (checked, matched)


def test_windows_results(native):
    """The Windows round's rigs: SkinTokens' mine_rodin (52 bones) and UniRig's (45 bones)."""
    st = GlbRig(native / "output/st_fresh/mine_rodin_a/rigged.glb")
    found = names.match(st.parents)
    if found is not None:
        assert names.placement_warnings(st.joints, found[1]) == []
    ur = GlbRig(native / "output/ur_full/mine_rodin_a/rigged.glb")
    assert len(ur.parents) == 45 and names.match(ur.parents) is None

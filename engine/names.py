"""Standard (Mixamo) bone names for a made skeleton, by its parent structure.

Neither model names its bones. But SkinTokens' default class often makes exactly the Mixamo layout
(Docker round, work-B): 22 bones in the order of the models' own configs/skeleton/mixamo.yaml, or
52 bones depth first with each hand's fingers right after that hand. When the parent array equals
one of these templates exactly, the bones get the template's names; otherwise nothing is renamed.

The names are the ones in both models' configs/skeleton/mixamo.yaml (`mixamorig:Hips` ...); a test
checks them against that file. The placement check (is the left hand on the character's left?) is
reported only, never used to decide: the rule is the parent structure.
"""
import numpy as np

PREFIX = "mixamorig:"
BODY = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]
BODY_PARENT = {"Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2",
               "Head": "Neck", "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm",
               "LeftHand": "LeftForeArm", "RightShoulder": "Spine2", "RightArm": "RightShoulder",
               "RightForeArm": "RightArm", "RightHand": "RightForeArm", "LeftUpLeg": "Hips",
               "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot", "RightUpLeg": "Hips",
               "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot"}
# Finger order as the yaml lists it: the left hand starts with the thumb, the right with the index.
HAND = []
for _side, _order in (("Left", ["Thumb", "Index", "Middle", "Ring", "Pinky"]),
                      ("Right", ["Index", "Thumb", "Middle", "Ring", "Pinky"])):
    for _finger in _order:
        HAND += [f"{_side}Hand{_finger}{k}" for k in (1, 2, 3)]
HAND_PARENT = {n: (n[:-1] + str(int(n[-1]) - 1) if n[-1] != "1" else n.split("Hand")[0] + "Hand") for n in HAND}
PARENT = dict(BODY_PARENT, **HAND_PARENT)


def template(layout):
    """Bare names (no prefix) and parents (-1 for the root) of one layout."""
    if layout == "body22":
        names = list(BODY)
    elif layout == "full52-yaml":
        names = BODY + HAND
    elif layout == "full52-dfs":
        left, right = HAND[:15], HAND[15:]
        names = BODY[:10] + left + BODY[10:14] + right + BODY[14:]
    else:
        raise ValueError(layout)
    return names, [-1 if PARENT[n] is None else names.index(PARENT[n]) for n in names]


LAYOUTS = {22: ("body22",), 52: ("full52-yaml", "full52-dfs")}


def match(parents):
    """The layout whose parent array equals `parents` exactly, as (layout, prefixed names), or None."""
    parents = [-1 if p is None or int(p) < 0 else int(p) for p in parents]
    for layout in LAYOUTS.get(len(parents), ()):
        names, want = template(layout)
        if parents == want:
            return layout, [PREFIX + n for n in names]
    return None


def placement_warnings(joints, names):
    """Checks from work-B. Up is head minus feet, forward is toes minus feet, so the frame's own axes
    don't matter. Returns what looks wrong (empty when every bone sits where its name says)."""
    joints = np.asarray(joints, dtype=np.float64)
    p = {n.split(":")[-1]: joints[i] for i, n in enumerate(names)}
    try:
        hips, head = p["Hips"], p["Head"]
        lfoot, rfoot, ltoe, rtoe = p["LeftFoot"], p["RightFoot"], p["LeftToeBase"], p["RightToeBase"]
        lhand, rhand = p["LeftHand"], p["RightHand"]
    except KeyError:
        return ["not a standard layout"]
    feet = (lfoot + rfoot) / 2
    up = head - feet
    if np.linalg.norm(up) == 0:
        return ["head and feet in the same place"]
    up /= np.linalg.norm(up)
    forward = (ltoe + rtoe) / 2 - feet
    forward -= up * forward.dot(up)
    if np.linalg.norm(forward) == 0:
        return ["no forward direction from the toes"]
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)

    def side(v):
        return float((v - hips).dot(left))

    def height(v):
        return float((v - feet).dot(up))

    fails = []
    if not height(feet) < height(hips) < height(head):
        fails.append("hips not between feet and head")
    if abs(side(hips)) > 0.25 * abs(side(lhand) - side(rhand)):
        fails.append("hips off centre")
    if not side(lhand) > 0 > side(rhand):
        fails.append("hands on the wrong sides")
    if not side(lfoot) > 0 > side(rfoot):
        fails.append("feet on the wrong sides")
    if not (height(lhand) > height(lfoot) and height(rhand) > height(rfoot)):
        fails.append("hands below feet")
    if "LeftHandIndex1" in p:
        lindex, rindex = p["LeftHandIndex1"], p["RightHandIndex1"]
        if np.linalg.norm(lindex - lhand) > np.linalg.norm(lindex - rhand):
            fails.append("left fingers nearer the right hand")
        if np.linalg.norm(rindex - rhand) > np.linalg.norm(rindex - lhand):
            fails.append("right fingers nearer the left hand")
    return fails


def decide(parents, joints, asked):
    """The standard names decision for a made skeleton. Returns (names or None, report dict)."""
    if not asked:
        return None, {"asked": False, "matched": False, "layout": None, "placement_warnings": []}
    found = match(parents)
    if found is None:
        return None, {"asked": True, "matched": False, "layout": None, "placement_warnings": []}
    layout, names = found
    return names, {"asked": True, "matched": True, "layout": layout,
                   "placement_warnings": placement_warnings(joints, names)}

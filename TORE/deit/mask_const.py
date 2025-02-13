from functools import reduce

import numpy as np

def _generate_masks(xs, ys):
    masks = []
    W, H = sum(xs), sum(ys)
    y_offset = 0
    for y in ys:
        x_offset = 0
        for x in xs:
            mask = np.zeros([H, W], dtype=bool)
            mask[y_offset:y_offset+y, x_offset:x_offset+x] = 1
            masks.append(mask)
            x_offset += x
        y_offset += y
    return masks

def _generate_ids(xs, ys):
    ids = []
    W = sum(xs)
    y_offset = 0
    for y in ys:
        x_offset = 0
        for x in xs:
            id = [x_+y_*W for y_ in range(y_offset, y_offset+y) for x_ in range(x_offset, x_offset+x)]
            ids.append(id)
            x_offset += x
        y_offset += y
    return ids


def division_masks_from_spec(specs):
    ret = {}
    for k, v in specs.items():
        ret[k] = [_generate_masks(**v)]
        v_rot = {"xs": v["ys"], "ys": v["xs"]}
        ret[k].append(_generate_masks(**v_rot))
    return ret


def division_masks_from_spec_ret(specs):
    ret = {}
    for k, v in specs.items():
        ret[k] = [_generate_masks(**v)]
    return ret

def division_ids_from_spec(specs):
    ret = {}
    for k, v in specs.items():
        ret[k] = [_generate_ids(**v)]
        v_rot = {"xs": v["ys"], "ys": v["xs"]}
        ret[k].append(_generate_ids(**v_rot))
    return ret


DIVISION_SPECS_14_14 = {
    1: {"xs": [14], "ys": [14]},
    2: {"xs": [14], "ys": [7, 7]},
    4: {"xs": [7, 7], "ys": [7, 7]},
    8: {"xs": [7, 7], "ys": [4, 3, 3, 4]},
    16: {"xs": [4, 3, 3, 4], "ys": [4, 3, 3, 4]},
    3: {"xs": [14], "ys": [5, 4, 5]},
    6: {"xs": [7, 7], "ys": [5, 4, 5]},
    9: {"xs": [5, 4, 5], "ys": [5, 4, 5]},
    12: {"xs": [4, 3, 3, 4], "ys": [5, 4, 5]},
}

DIVISION_SPECS_28_28 = {}
for key, value in DIVISION_SPECS_14_14.items():
    DIVISION_SPECS_28_28[key] = {
        "xs": [x * 2 for x in value["xs"]],
        "ys": [y * 2 for y in value["ys"]],
    }

DIVISION_SPECS_56_56 = {}
for key, value in DIVISION_SPECS_14_14.items():
    DIVISION_SPECS_56_56[key] = {
        "xs": [x * 4 for x in value["xs"]],
        "ys": [y * 4 for y in value["ys"]],
    }

DIVISION_SPECS_12_12 = {
    1: {"xs": [12], "ys": [12]},
    2: {"xs": [12], "ys": [6, 6]},
    4: {"xs": [6, 6], "ys": [6, 6]},
    8: {"xs": [6, 6], "ys": [3, 3, 3, 3]},
    16: {"xs": [3, 3, 3, 3], "ys": [3, 3, 3, 3]},
    3: {"xs": [12], "ys": [4, 4, 4]},
    6: {"xs": [6, 6], "ys": [4, 4, 4]},
    9: {"xs": [4, 4, 4], "ys": [4, 4, 4]},
    12: {"xs": [3, 3, 3, 3], "ys": [4, 4, 4]},
}

DIVISION_SPECS_30_30 = {
    16: {"xs": [8, 7, 7, 8], "ys": [8, 7, 7, 8]},
}

DIVISION_SPECS_60_60 = {
    16: {"xs": [16, 14, 14, 16], "ys": [16, 14, 14, 16]},
}

DIVISION_SPECS_832 = {
    16: {"xs": [26, 26, 26, 26], "ys": [16, 14, 14, 16]},
}

DIVISION_SPECS_896 = {
    16: {"xs": [28, 28, 28, 28], "ys": [16, 14, 14, 16]},
}

DIVISION_SPECS_1152 = {
    16: {"xs": [36, 36, 36, 36], "ys": [16, 14, 14, 16]},
}

DIVISION_MASKS = {
    12: division_masks_from_spec(DIVISION_SPECS_12_12),
    14: division_masks_from_spec(DIVISION_SPECS_14_14),
    28: division_masks_from_spec(DIVISION_SPECS_28_28),
    30: division_masks_from_spec(DIVISION_SPECS_30_30),
    60: division_masks_from_spec(DIVISION_SPECS_60_60)
}

DIVISION_IDS = {
    12: division_ids_from_spec(DIVISION_SPECS_12_12),
    14: division_ids_from_spec(DIVISION_SPECS_14_14),
    28: division_ids_from_spec(DIVISION_SPECS_28_28)
}


import random
def sample_masks(division_masks, M):
    m_id = random.randint(0, len(division_masks[M]) - 1)
    masks = division_masks[M][m_id]
    return masks

def get_division_masks_for_model(model):
    assert model.patch_embed.img_size[0] == model.patch_embed.img_size[1]
    assert model.patch_embed.patch_size[0] == model.patch_embed.patch_size[1]
    division_masks = DIVISION_MASKS[model.patch_embed.img_size[0] // model.patch_embed.patch_size[0]]
    return division_masks

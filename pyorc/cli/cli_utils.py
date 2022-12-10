import os
import json

def parse_json(ctx, param, value):
    if value is None:
        return {}
    if os.path.isfile(value):
        with open(value, "r") as f:
            kwargs = json.load(f)
    else:
        if value.strip("{").startswith("'"):
            value = value.replace("'", '"')
        try:
            kwargs = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f'Could not decode JSON "{value}"')
    return kwargs

def parse_corners(ctx, param, value):
    # check if list contains lists of 2 values
    print(value)
    print(json.loads(value))
    corners = json.loads(value)
    assert(len(corners)==4), "--corners must contain a list of lists with exactly 4 points"
    for n, val in enumerate(corners):
        assert(isinstance(val, list)), f"--corners value {n} is not a list {val}"
        assert(len(val)==2), f"--corners value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return [[int(x), int(y)] for x, y in corners]

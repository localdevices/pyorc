import click
import os
import json

def parse_json(ctx, param, value):
    if value is None:
        return None
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
    if value is None:
        return None
    # check if list contains lists of 2 values
    print(value)
    print(json.loads(value))
    corners = json.loads(value)
    assert(len(corners)==4), "--corners must contain a list of lists with exactly 4 points"
    for n, val in enumerate(corners):
        assert(isinstance(val, list)), f"--corners value {n} is not a list {val}"
        assert(len(val)==2), f"--corners value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return [[int(x), int(y)] for x, y in corners]


def validate_file(ctx, param, value):
    if value is not None:
        if not(os.path.isfile(value)):
            raise click.FileError(f"{value}")
        return value


def parse_src(ctx, param, value):
    value = parse_json(ctx, param, value)
    if value is not None:
        # check if at least 4 of 2
        assert(len(value)>=4), "--src must contain a list of lists [column, row] with at least 4 points"
        for n, val in enumerate(value):
            assert(isinstance(val, list)), f"--src value {n} is not a list {val}"
            assert(len(val) == 2), f"--src value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return value

def parse_dst(ctx, param, value):
    value = parse_json(ctx, param, value)
    if value is not None:
        if len(value) == 4:
            # assume [x, y] pairs are provided
            len_points = 2
        elif len(value) < 6:
            raise click.UsageError(f"--dst must contain at least 4 with [x, y] or 6 with [x, y, z] points, contains {len(value)}.")
        else:
            len_points = 3
        for n, val in enumerate(value):
            assert(isinstance(val, list)), f"--dst value {n} is not a list {val}"
            assert(len(val) == len_points), f"--src value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return value


def parse_str_num(ctx, param, value):
    if value is not None:
        try:
            float(value)
        except:
            return value
        if value.isnumeric():
            return int(value)
        else:
            return float(value)

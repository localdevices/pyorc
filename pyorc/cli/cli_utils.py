import click
import os
import json
import pyorc
import yaml

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
    assert(len(corners) == 4), "--corners must contain a list of lists with exactly 4 points"
    for n, val in enumerate(corners):
        assert(isinstance(val, list)), f"--corners value {n} is not a list {val}"
        assert(len(val) == 2), f"--corners value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return [[int(x), int(y)] for x, y in corners]


def validate_file(ctx, param, value):
    if value is not None:
        if not(os.path.isfile(value)):
            raise click.FileError(f"{value}")
        return value

def validate_dir(ctx, param, value):
    if not(os.path.isdir(value)):
        os.makedirs(value)
    return value
def parse_recipe(ctx, param, recipe_file):
    """
    Read and validate entire recipe from top to bottom, add compulsory classes where needed

    Parameters
    ----------
    recipe_file : str,
        file containing .yml formatted recipe

    Returns
    -------
    recipe : dict,
        dictionary with entire recipe for running processing
    """
    with open(recipe_file, "r") as f:
        body = f.read()
    recipe = yaml.load(body, Loader=yaml.FullLoader)

    valid_classes = ["video", "frames", "velocimetry", "transect"]  # allowed classes
    required_classes = ["video", "frames", "velocimetry"]  # mandatory classes (if not present, these are added)
    check_args = ["video", "frames"]  # check if arguments to underlying methods are valid in these classes
    process_methods = ["write"]  # methods that are specifically needed within process steps and not part of pyorc class methods
    for k in recipe:
        if k not in valid_classes:
            raise ValueError(f"key '{k}' is not allowed, must be one of {valid_classes}")
        if k in check_args:
            # loop through all methods and check if their inputs are valid
            cls = getattr(pyorc, k.capitalize())
            for m in recipe[k]:
                if m not in process_methods:
                    if (not hasattr(cls, m)):
                        raise ValueError(f"Class '{k.capitalize()}' does not have a method or property '{m}'")
                    method = getattr(cls, m)
                    # find valid args to method
                    if hasattr(method, "__call__"):
                        valid_args = method.__code__.co_varnames
                        if recipe[k][m] is None:
                            # replace for empty dict
                            recipe[k][m] = {}
                        for arg in recipe[k][m]:
                            if not(arg in valid_args):
                                raise ValueError(f"Method '{k.capitalize()}.{m}' does not have input argument '{arg}', must be one of {valid_args}")
    # add empty dicts for missing but compulsory classes
    for _c in required_classes:
        if _c not in recipe:
            # add empties for compulsory recipe components
            recipe[_c] = {}
    print(recipe)
    return recipe


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

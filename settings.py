"""Typed configuration overrides from literal assignments and command-line options."""

import argparse
import ast
from dataclasses import asdict, fields
from pathlib import Path


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_settings(config_type, arguments=None):
    defaults = config_type()
    parser = argparse.ArgumentParser(description=config_type.__doc__)
    parser.add_argument("configs", nargs="*", help="Python files containing literal setting assignments")
    for field in fields(defaults):
        parser.add_argument(f"--{field.name}", type=parse_value, default=argparse.SUPPRESS)
    options = vars(parser.parse_args(arguments))
    values = asdict(defaults)

    def update(name, value):
        if name not in values:
            parser.error(f"Unknown setting: {name}")
        expected = type(values[name])
        if expected is float and type(value) is int:
            value = float(value)
        if type(value) is not expected:
            parser.error(f"{name} requires {expected.__name__}, received {value!r}")
        values[name] = value

    for filename in options.pop("configs"):
        try:
            statements = ast.parse(Path(filename).read_text(encoding="utf-8"), filename=filename).body
            for statement in statements:
                if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                        or not isinstance(statement.targets[0], ast.Name)):
                    parser.error(f"{filename}: only literal assignments are supported")
                update(statement.targets[0].id, ast.literal_eval(statement.value))
        except (OSError, SyntaxError, ValueError) as error:
            parser.error(str(error))
    for name, value in options.items():
        update(name, value)
    return config_type(**values)

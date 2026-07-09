"""
Configurable weight transforms (a.k.a. weight functions / shaping functions).

Replaces the static-weight model

    y = sum(w_i * x_i)

with a transform-based model

    y = sum(f_i(x_i))

where each f_i is a small, named, composable, and inspectable "knob"
(clamp, amplify, compress, threshold, saturate, etc.) instead of a
single scalar weight.

TRANSFORM_SPECS describes each knob's parameters (label, default, step)
so a UI can render the right controls without hardcoding anything.

This module is domain-agnostic: it has no notion of "words", "courses",
or any particular dataset. It only knows about numeric channels and the
knobs applied to them. Anything dataset-specific lives in
dataset_loader.py / the per-dataset config.json files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np

Number = Union[float, int, np.ndarray]


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


@dataclass
class Transform:
    """A named, callable weight transform with a readable repr."""

    name: str
    fn: Callable[[Number], Number]
    params: dict = field(default_factory=dict)

    def __call__(self, x: Number) -> Number:
        return self.fn(x)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"

    def to_latex(self, var: str) -> str:
        """Returns the LaTeX mathematical representation for this transform."""
        name = self.name
        p = self.params
        if name == "linear":
            w = p.get("w", 1.0)
            return f"{w:.1f} \\\\cdot {var}"
        elif name == "clamp":
            min_v = p.get("min_v")
            max_v = p.get("max_v")
            scale = p.get("scale", 1.0)
            if min_v is not None and max_v is not None:
                return f"{scale:.2f} \\\\cdot \\\\min(\\\\max({var}, {min_v:.2f}), {max_v:.2f})"
            elif min_v is not None:
                return f"{scale:.2f} \\\\cdot \\\\max({var}, {min_v:.2f})"
            elif max_v is not None:
                return f"{scale:.2f} \\\\cdot \\\\min({var}, {max_v:.2f})"
            else:
                return f"{scale:.2f} \\\\cdot {var}"
        elif name == "amplify":
            power = p.get("power", 2.0)
            scale = p.get("scale", 1.0)
            return f"{scale:.2f} \\\\cdot \\\\operatorname{{sgn}}({var}) \\\\cdot |{var}|^{{{power:.2f}}}"
        elif name == "log_compress":
            scale = p.get("scale", 1.0)
            return f"{scale:.2f} \\\\cdot \\\\ln(1 + \\\\max(0, {var}))"
        elif name == "sigmoid":
            center = p.get("center", 50.0)
            sharpness = p.get("sharpness", 0.1)
            scale = p.get("scale", 1.0)
            return f"\\\\frac{{{scale:.2f}}}{{1 + e^{{-{sharpness:.3f} \\\\cdot ({var} - {center:.2f})}}}}"
        elif name == "threshold":
            level = p.get("level", 0.0)
            below = p.get("below", 0.0)
            above = p.get("above", 1.0)
            return f"\\\\begin{{cases}} {below:.2f} & {var} < {level:.2f} \\\\\\\\ {above:.2f} & {var} \\\\ge {level:.2f} \\\\end{{cases}}"
        elif name == "passthrough":
            return var
        elif name == "compose":
            steps_str = p.get("steps", "")
            return f"\\\\operatorname{{{steps_str}}}({var})"
        else:
            return f"\\\\operatorname{{{name}}}({var})"

    def to_python(self, var: str) -> str:
        """Returns the evaluatable NumPy Python code string for this transform."""
        name = self.name
        p = self.params
        if name == "linear":
            w = p.get("w", 1.0)
            return f"{w:.2f} * {var}"
        elif name == "clamp":
            min_v = p.get("min_v")
            max_v = p.get("max_v")
            scale = p.get("scale", 1.0)
            if min_v is not None and max_v is not None:
                return f"{scale:.2f} * np.clip({var}, {min_v:.2f}, {max_v:.2f})"
            elif min_v is not None:
                return f"{scale:.2f} * np.maximum({var}, {min_v:.2f})"
            elif max_v is not None:
                return f"{scale:.2f} * np.minimum({var}, {max_v:.2f})"
            else:
                return f"{scale:.2f} * {var}"
        elif name == "amplify":
            power = p.get("power", 2.0)
            scale = p.get("scale", 1.0)
            return f"{scale:.2f} * np.sign({var}) * (np.abs({var}) ** {power:.2f})"
        elif name == "log_compress":
            scale = p.get("scale", 1.0)
            return f"{scale:.2f} * np.log1p(np.maximum(0.0, {var}))"
        elif name == "sigmoid":
            center = p.get("center", 50.0)
            sharpness = p.get("sharpness", 0.1)
            scale = p.get("scale", 1.0)
            return f"{scale:.2f} / (1.0 + np.exp(-{sharpness:.3f} * ({var} - {center:.2f})))"
        elif name == "threshold":
            level = p.get("level", 0.0)
            below = p.get("below", 0.0)
            above = p.get("above", 1.0)
            return f"np.where({var} < {level:.2f}, {below:.2f}, {above:.2f})"
        elif name == "passthrough":
            return var
        elif name == "compose":
            steps_str = p.get("steps", "")
            return f"{steps_str}({var})"
        else:
            return f"{name}({var})"


# ---------------------------------------------------------------------------
# Transform factories ("knobs")
# ---------------------------------------------------------------------------


def linear(w: float = 1.0) -> Transform:
    return Transform("linear", lambda x: w * x, {"w": w})


def clamp(
    min_v: float | None = None, max_v: float | None = None, scale: float = 1.0
) -> Transform:
    def f(x):
        if min_v is None and max_v is None:
            return scale * x
        return scale * np.clip(x, min_v, max_v)

    return Transform("clamp", f, {"min_v": min_v, "max_v": max_v, "scale": scale})


def amplify(power: float = 2.0, scale: float = 1.0) -> Transform:
    def f(x):
        x = np.asarray(x, dtype=float)
        return scale * np.sign(x) * np.abs(x) ** power

    return Transform("amplify", f, {"power": power, "scale": scale})


def log_compress(scale: float = 1.0) -> Transform:
    def f(x):
        x = np.asarray(x, dtype=float)
        return scale * np.log1p(np.maximum(0.0, x))

    return Transform("log_compress", f, {"scale": scale})


def sigmoid(
    center: float = 50, sharpness: float = 0.1, scale: float = 1.0
) -> Transform:
    def f(x):
        x = np.asarray(x, dtype=float)
        return scale / (1 + np.exp(-sharpness * (x - center)))

    return Transform(
        "sigmoid", f, {"center": center, "sharpness": sharpness, "scale": scale}
    )


def threshold(level: float = 0, below: float = 0.0, above: float = 1.0) -> Transform:
    """Step function: `below` if x < level else `above`."""

    def f(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < level, below, above)

    return Transform("threshold", f, {"level": level, "below": below, "above": above})


def passthrough() -> Transform:
    return Transform("passthrough", lambda x: x, {})


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(*steps: Callable) -> Transform:
    """Chain transforms left-to-right into a single knob."""
    names = " -> ".join(getattr(s, "name", getattr(s, "__name__", "fn")) for s in steps)

    def f(x):
        for step in steps:
            x = step(x)
        return x

    return Transform("compose", f, {"steps": names})


# ---------------------------------------------------------------------------
# Config-driven registry (lets knobs be defined as data, not code)
# ---------------------------------------------------------------------------

REGISTRY = {
    "linear": linear,
    "clamp": clamp,
    "amplify": amplify,
    "log_compress": log_compress,
    "sigmoid": sigmoid,
    "threshold": threshold,
    "passthrough": passthrough,
}


# UI/API-facing metadata: label, help text, and per-parameter widget hints.
# `nullable: true` means the param accepts an empty value (mapped to None).
TRANSFORM_SPECS = {
    "linear": {
        "label": "Linear",
        "description": "w × x",
        "params": {
            "w": {"label": "weight", "default": 1.0, "step": 0.1},
        },
    },
    "clamp": {
        "label": "Clamp",
        "description": "Cap x to [min, max], then scale",
        "params": {
            "min_v": {"label": "min", "default": None, "step": 0.05, "nullable": True},
            "max_v": {"label": "max", "default": 1.0, "step": 0.05, "nullable": True},
            "scale": {"label": "scale", "default": 1.0, "step": 0.1},
        },
    },
    "amplify": {
        "label": "Amplify",
        "description": "scale × sign(x) × |x|^power",
        "params": {
            "power": {"label": "power", "default": 2.0, "step": 0.1},
            "scale": {"label": "scale", "default": 1.0, "step": 0.1},
        },
    },
    "log_compress": {
        "label": "Log compress",
        "description": "scale × log(1 + max(0, x))",
        "params": {
            "scale": {"label": "scale", "default": 1.0, "step": 0.1},
        },
    },
    "sigmoid": {
        "label": "Sigmoid",
        "description": "Smooth saturation curve",
        "params": {
            "center": {"label": "center", "default": 0.5, "step": 0.05},
            "sharpness": {"label": "sharpness", "default": 10.0, "step": 0.5},
            "scale": {"label": "scale", "default": 1.0, "step": 0.1},
        },
    },
    "threshold": {
        "label": "Threshold",
        "description": "below/above a level (step function)",
        "params": {
            "level": {"label": "level", "default": 0.5, "step": 0.05},
            "below": {"label": "below value", "default": 0.0, "step": 0.1},
            "above": {"label": "above value", "default": 1.0, "step": 0.1},
        },
    },
    "passthrough": {
        "label": "Passthrough",
        "description": "No change",
        "params": {},
    },
}


def from_config(spec: dict) -> Transform:
    """Build a Transform from a dict, e.g. {"type": "clamp", "max_v": 0.6}.

    This makes knobs JSON-serializable, which is useful for grid search,
    saved presets, learned/optimized parameters, and web APIs.
    """
    spec = dict(spec)
    kind = spec.pop("type", None)
    try:
        factory = REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown transform type: {kind!r}") from None

    # Drop any keys the factory doesn't accept (e.g. stray UI fields)
    valid_params = TRANSFORM_SPECS[kind]["params"].keys()
    kwargs = {k: v for k, v in spec.items() if k in valid_params}
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def weighted_sum(inputs: Sequence[Number], transforms: Sequence[Callable]):
    if len(inputs) != len(transforms):
        raise ValueError(
            f"inputs ({len(inputs)}) and transforms ({len(transforms)}) must match"
        )

    contributions = [float(t(x)) for x, t in zip(inputs, transforms)]
    total = sum(contributions)
    return total, contributions


def vectorized_weighted_sum(
    inputs: Sequence[np.ndarray], transforms: Sequence[Callable]
):
    """Vectorized calculation of pipeline transforms without float scalar-casting."""
    if len(inputs) != len(transforms):
        raise ValueError(
            f"inputs ({len(inputs)}) and transforms ({len(transforms)}) must match"
        )

    contributions = [t(x) for x, t in zip(inputs, transforms)]
    total = sum(contributions)
    return total, contributions


def compile_pipeline_formulas(
    knobs: Sequence[Transform],
    latex_vars: Sequence[str] | None = None,
    python_vars: Sequence[str] | None = None,
) -> dict[str, str]:
    """Compiles any number of channel transforms into a single summed LaTeX
    and Python formula: y = f_1(x_1) + f_2(x_2) + ... + f_n(x_n).

    Unlike the fixed-3-channel version this replaces, this works for any
    number of knobs, in the order given. Pass `latex_vars` / `python_vars`
    to label each term with something more meaningful than x_1, x_2... (e.g.
    the dataset's own column names).
    """
    n = len(knobs)
    if n == 0:
        return {"latex": "y = 0", "python": "y = 0"}

    if latex_vars is None:
        latex_vars = [f"x_{{{i + 1}}}" for i in range(n)]
    if python_vars is None:
        python_vars = [f"x{i + 1}" for i in range(n)]

    if len(latex_vars) != n or len(python_vars) != n:
        raise ValueError("latex_vars/python_vars must match the number of knobs")

    latex_terms = [k.to_latex(v) for k, v in zip(knobs, latex_vars)]
    python_terms = [k.to_python(v) for k, v in zip(knobs, python_vars)]

    latex_str = "y = " + " + ".join(f"\\\\left( {t} \\\\right)" for t in latex_terms)
    python_str = "y = " + " + ".join(f"({t})" for t in python_terms)

    return {"latex": latex_str, "python": python_str}


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    X = [100, 50, 20]

    knobs = [
        clamp(max_v=60),
        amplify(power=1.5, scale=0.2),
        log_compress(scale=10),
    ]

    score, contributions = weighted_sum(X, knobs)

    for x, knob, c in zip(X, knobs, contributions):
        print(f"{x:>5}  --  {knob!r:35}  ->  {c:.2f}")
    print(f"\ntotal: {score:.2f}")

    print(compile_pipeline_formulas(knobs)["latex"])

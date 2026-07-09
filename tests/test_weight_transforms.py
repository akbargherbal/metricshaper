import numpy as np
import pytest

import weight_transforms as wt

# ---------------------------------------------------------------------------
# Factories: scalar + ndarray behavior
# ---------------------------------------------------------------------------


class TestLinear:
    def test_default_weight(self):
        t = wt.linear()
        assert t(4) == 4.0

    def test_explicit_weight_scalar(self):
        t = wt.linear(w=2.5)
        assert t(4) == 10.0

    def test_array_input(self):
        t = wt.linear(w=2.0)
        np.testing.assert_allclose(t(np.array([1, 2, 3])), [2, 4, 6])


class TestClamp:
    def test_min_only(self):
        t = wt.clamp(min_v=2.0)
        assert t(1.0) == 2.0
        assert t(5.0) == 5.0

    def test_max_only(self):
        t = wt.clamp(max_v=3.0)
        assert t(5.0) == 3.0
        assert t(1.0) == 1.0

    def test_both(self):
        t = wt.clamp(min_v=1.0, max_v=3.0)
        assert t(-5.0) == 1.0
        assert t(5.0) == 3.0
        assert t(2.0) == 2.0

    def test_neither_falls_back_to_scale(self):
        t = wt.clamp(scale=2.0)
        assert t(3.0) == 6.0

    def test_array_input(self):
        t = wt.clamp(min_v=0.0, max_v=1.0)
        np.testing.assert_allclose(t(np.array([-1, 0.5, 2])), [0, 0.5, 1])


class TestAmplify:
    def test_default(self):
        t = wt.amplify()
        assert t(2.0) == pytest.approx(4.0)

    def test_negative_preserves_sign(self):
        t = wt.amplify(power=2.0)
        assert t(-2.0) == pytest.approx(-4.0)

    def test_scale(self):
        t = wt.amplify(power=1.0, scale=0.5)
        assert t(4.0) == pytest.approx(2.0)

    def test_array_input(self):
        t = wt.amplify(power=2.0)
        np.testing.assert_allclose(t(np.array([-2, 0, 2])), [-4, 0, 4])


class TestLogCompress:
    def test_default(self):
        t = wt.log_compress()
        assert t(0.0) == pytest.approx(0.0)

    def test_scale(self):
        t = wt.log_compress(scale=2.0)
        assert t(0.0) == pytest.approx(0.0)
        assert t(np.e - 1) == pytest.approx(2.0)

    def test_negative_clamped_to_zero(self):
        t = wt.log_compress()
        assert t(-5.0) == pytest.approx(0.0)

    def test_array_input(self):
        t = wt.log_compress()
        result = t(np.array([-1, 0, np.e - 1]))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-9)


class TestSigmoid:
    def test_at_center_returns_half_scale(self):
        t = wt.sigmoid(center=50.0, sharpness=0.1, scale=1.0)
        assert t(50.0) == pytest.approx(0.5)

    def test_custom_params(self):
        t = wt.sigmoid(center=0.0, sharpness=1.0, scale=2.0)
        assert t(0.0) == pytest.approx(1.0)

    def test_array_input(self):
        t = wt.sigmoid(center=0.0, sharpness=1.0, scale=1.0)
        result = t(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5])


class TestThreshold:
    def test_below_level(self):
        t = wt.threshold(level=5.0, below=0.0, above=1.0)
        assert t(4.0) == 0.0

    def test_above_level(self):
        t = wt.threshold(level=5.0, below=0.0, above=1.0)
        assert t(6.0) == 1.0

    def test_exactly_at_level_is_above(self):
        # boundary is inclusive per np.where(x < level, ...) -> equal goes to "above"
        t = wt.threshold(level=5.0, below=0.0, above=1.0)
        assert t(5.0) == 1.0

    def test_array_input(self):
        t = wt.threshold(level=0.0, below=-1.0, above=1.0)
        np.testing.assert_allclose(t(np.array([-1, 0, 1])), [-1, 1, 1])


class TestPassthrough:
    def test_scalar(self):
        t = wt.passthrough()
        assert t(7) == 7

    def test_array(self):
        t = wt.passthrough()
        arr = np.array([1, 2, 3])
        np.testing.assert_array_equal(t(arr), arr)


# ---------------------------------------------------------------------------
# Transform.__call__ / __repr__
# ---------------------------------------------------------------------------


def test_transform_call_delegates_to_fn():
    t = wt.Transform("id", lambda x: x * 2, {})
    assert t(5) == 10


def test_transform_repr_contains_name_and_params():
    t = wt.linear(w=3.0)
    r = repr(t)
    assert r == "linear(w=3.0)"


def test_transform_repr_empty_params():
    t = wt.passthrough()
    assert repr(t) == "passthrough()"


# ---------------------------------------------------------------------------
# Transform.to_latex / to_python — one case per branch, exact strings
# ---------------------------------------------------------------------------

# NOTE: expected LaTeX strings are built with chr(92) (rather than typed
# backslash-escape sequences) to avoid any ambiguity about how many literal
# backslash characters end up in the string. This module's to_latex output
# uses TWO literal backslash characters per LaTeX command (e.g. "\\cdot"),
# so BS below is deliberately doubled per command.
BS = chr(92)  # a single literal backslash character
D = BS + BS  # the double-backslash sequence this module actually emits

LATEX_CASES = [
    ("linear", {"w": 1.0}, f"1.0 {D}cdot x"),
    (
        "clamp",
        {"min_v": 1.0, "max_v": 5.0, "scale": 2.0},
        f"2.00 {D}cdot {D}min({D}max(x, 1.00), 5.00)",
    ),
    ("clamp", {"min_v": 1.0}, f"1.00 {D}cdot {D}max(x, 1.00)"),
    ("clamp", {"max_v": 5.0}, f"1.00 {D}cdot {D}min(x, 5.00)"),
    ("clamp", {}, f"1.00 {D}cdot x"),
    (
        "amplify",
        {"power": 3.0, "scale": 2.0},
        f"2.00 {D}cdot {D}operatorname{{sgn}}(x) {D}cdot |x|^{{3.00}}",
    ),
    ("log_compress", {"scale": 2.0}, f"2.00 {D}cdot {D}ln(1 + {D}max(0, x))"),
    (
        "sigmoid",
        {"center": 1.0, "sharpness": 2.0, "scale": 3.0},
        f"{D}frac{{3.00}}{{1 + e^{{-2.000 {D}cdot (x - 1.00)}}}}",
    ),
    (
        "threshold",
        {"level": 1.0, "below": 0.0, "above": 1.0},
        f"{D}begin{{cases}} 0.00 & x < 1.00 {D}{D} 1.00 & x {D}ge 1.00 {D}end{{cases}}",
    ),
    ("passthrough", {}, "x"),
]

PYTHON_CASES = [
    ("linear", {"w": 1.0}, "1.00 * x"),
    (
        "clamp",
        {"min_v": 1.0, "max_v": 5.0, "scale": 2.0},
        "2.00 * np.clip(x, 1.00, 5.00)",
    ),
    ("clamp", {"min_v": 1.0}, "1.00 * np.maximum(x, 1.00)"),
    ("clamp", {"max_v": 5.0}, "1.00 * np.minimum(x, 5.00)"),
    ("clamp", {}, "1.00 * x"),
    (
        "amplify",
        {"power": 3.0, "scale": 2.0},
        "2.00 * np.sign(x) * (np.abs(x) ** 3.00)",
    ),
    ("log_compress", {"scale": 2.0}, "2.00 * np.log1p(np.maximum(0.0, x))"),
    (
        "sigmoid",
        {"center": 1.0, "sharpness": 2.0, "scale": 3.0},
        "3.00 / (1.0 + np.exp(-2.000 * (x - 1.00)))",
    ),
    (
        "threshold",
        {"level": 1.0, "below": 0.0, "above": 1.0},
        "np.where(x < 1.00, 0.00, 1.00)",
    ),
    ("passthrough", {}, "x"),
]


@pytest.mark.parametrize("name,kwargs,expected", LATEX_CASES)
def test_to_latex_exact(name, kwargs, expected):
    t = wt.REGISTRY[name](**kwargs)
    assert t.to_latex("x") == expected


@pytest.mark.parametrize("name,kwargs,expected", PYTHON_CASES)
def test_to_python_exact(name, kwargs, expected):
    t = wt.REGISTRY[name](**kwargs)
    assert t.to_python("x") == expected


def test_to_latex_compose_branch():
    t = wt.compose(wt.linear(2.0), wt.clamp(max_v=3.0))
    bs = chr(92) * 2
    assert t.to_latex("x") == f"{bs}operatorname{{linear -> clamp}}(x)"


def test_to_python_compose_branch():
    t = wt.compose(wt.linear(2.0), wt.clamp(max_v=3.0))
    assert t.to_python("x") == "linear -> clamp(x)"


def test_to_latex_unrecognized_name_fallback():
    t = wt.Transform("bogus", lambda x: x, {})
    bs = chr(92) * 2
    assert t.to_latex("x") == f"{bs}operatorname{{bogus}}(x)"


def test_to_python_unrecognized_name_fallback():
    t = wt.Transform("bogus", lambda x: x, {})
    assert t.to_python("x") == "bogus(x)"


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_compose_chains_two_steps():
    t = wt.compose(wt.linear(2.0), wt.clamp(max_v=3.0))
    assert t(10) == 3.0


def test_compose_chains_three_steps():
    t = wt.compose(
        wt.linear(2.0), wt.clamp(max_v=100.0), wt.amplify(power=1.0, scale=1.0)
    )
    assert t(4) == pytest.approx(8.0)


def test_compose_name_string():
    t = wt.compose(wt.linear(), wt.clamp(), wt.passthrough())
    assert t.params["steps"] == "linear -> clamp -> passthrough"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_valid_spec_builds_transform():
    t = wt.from_config({"type": "clamp", "max_v": 0.6})
    assert isinstance(t, wt.Transform)
    assert t.name == "clamp"
    assert t.params["max_v"] == 0.6


def test_from_config_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown transform type"):
        wt.from_config({"type": "not_a_real_transform"})


def test_from_config_drops_stray_keys():
    # "bogus_key" isn't a valid param for "linear" and must be silently dropped
    t = wt.from_config({"type": "linear", "w": 2.0, "bogus_key": "should be ignored"})
    assert t.params == {"w": 2.0}


# ---------------------------------------------------------------------------
# weighted_sum / vectorized_weighted_sum
# ---------------------------------------------------------------------------


def test_weighted_sum_total_and_contributions():
    inputs = [1.0, 2.0, 3.0]
    transforms = [wt.linear(2.0), wt.linear(1.0), wt.passthrough()]
    total, contributions = wt.weighted_sum(inputs, transforms)
    assert contributions == [2.0, 2.0, 3.0]
    assert total == 7.0


def test_weighted_sum_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="must match"):
        wt.weighted_sum([1.0, 2.0], [wt.linear()])


def test_vectorized_weighted_sum_returns_arrays():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    transforms = [wt.linear(2.0), wt.linear(1.0)]
    total, contributions = wt.vectorized_weighted_sum(inputs, transforms)
    assert isinstance(total, np.ndarray)
    assert all(isinstance(c, np.ndarray) for c in contributions)
    np.testing.assert_allclose(total, [5.0, 8.0])


def test_vectorized_weighted_sum_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="must match"):
        wt.vectorized_weighted_sum([np.array([1.0])], [])


# ---------------------------------------------------------------------------
# compile_pipeline_formulas
# ---------------------------------------------------------------------------


def test_compile_pipeline_formulas_zero_knobs():
    result = wt.compile_pipeline_formulas([])
    assert result == {"latex": "y = 0", "python": "y = 0"}


def test_compile_pipeline_formulas_default_var_names():
    knobs = [wt.linear(1.0), wt.linear(2.0)]
    result = wt.compile_pipeline_formulas(knobs)
    assert "x_{1}" in result["latex"]
    assert "x_{2}" in result["latex"]
    assert "x1" in result["python"]
    assert "x2" in result["python"]


def test_compile_pipeline_formulas_custom_var_names():
    knobs = [wt.linear(1.0), wt.linear(2.0)]
    bs = chr(92) * 2
    result = wt.compile_pipeline_formulas(
        knobs,
        latex_vars=[f"{bs}text{{rating}}", f"{bs}text{{price}}"],
        python_vars=["rating", "price"],
    )
    assert "rating" in result["python"]
    assert "price" in result["python"]
    assert "rating" in result["latex"]


def test_compile_pipeline_formulas_mismatched_latex_vars_raises():
    knobs = [wt.linear(1.0), wt.linear(2.0)]
    with pytest.raises(ValueError, match="must match"):
        wt.compile_pipeline_formulas(knobs, latex_vars=["only_one"])


def test_compile_pipeline_formulas_mismatched_python_vars_raises():
    knobs = [wt.linear(1.0), wt.linear(2.0)]
    with pytest.raises(ValueError, match="must match"):
        wt.compile_pipeline_formulas(knobs, python_vars=["only_one"])


# ---------------------------------------------------------------------------
# TRANSFORM_SPECS / REGISTRY consistency
# ---------------------------------------------------------------------------


def test_registry_keys_have_matching_specs():
    for key in wt.REGISTRY:
        assert key in wt.TRANSFORM_SPECS, f"{key} missing from TRANSFORM_SPECS"


def test_specs_keys_have_matching_registry_entries():
    for key in wt.TRANSFORM_SPECS:
        assert key in wt.REGISTRY, f"{key} missing from REGISTRY"

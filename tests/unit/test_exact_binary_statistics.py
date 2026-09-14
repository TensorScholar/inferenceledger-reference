from __future__ import annotations

from math import sqrt

import pytest

from inference_engine.benchmarking.statistics import (
    exact_binomial_lower_confidence_bound,
    exact_binomial_upper_confidence_bound,
)


def test_zero_events_matches_closed_form_one_sided_clopper_pearson_bound() -> None:
    confidence = 0.95
    trial_count = 100

    bound = exact_binomial_upper_confidence_bound(
        event_count=0,
        trial_count=trial_count,
        confidence_level=confidence,
    )

    expected = 1 - (1 - confidence) ** (1 / trial_count)
    assert bound == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_one_event_of_two_matches_exact_binomial_inversion() -> None:
    bound = exact_binomial_upper_confidence_bound(
        event_count=1,
        trial_count=2,
        confidence_level=0.95,
    )

    # P[X <= 1 | n=2,p] = 1 - p^2 = 0.05 at the one-sided 95% upper bound.
    assert bound == pytest.approx(sqrt(0.95), rel=1e-10, abs=1e-12)


def test_all_events_have_upper_bound_one() -> None:
    assert exact_binomial_upper_confidence_bound(
        event_count=8,
        trial_count=8,
        confidence_level=0.95,
    ) == 1.0


def test_upper_bound_is_at_least_observed_rate_and_decreases_with_more_trials() -> None:
    small = exact_binomial_upper_confidence_bound(
        event_count=1,
        trial_count=20,
        confidence_level=0.95,
    )
    large = exact_binomial_upper_confidence_bound(
        event_count=1,
        trial_count=200,
        confidence_level=0.95,
    )

    assert small >= 1 / 20
    assert large >= 1 / 200
    assert large < small


def test_exact_bound_rejects_invalid_counts_and_confidence() -> None:
    with pytest.raises(ValueError, match="positive"):
        exact_binomial_upper_confidence_bound(event_count=0, trial_count=0)
    with pytest.raises(ValueError, match="between"):
        exact_binomial_upper_confidence_bound(event_count=3, trial_count=2)
    with pytest.raises(ValueError, match="confidence_level"):
        exact_binomial_upper_confidence_bound(
            event_count=0,
            trial_count=10,
            confidence_level=1.0,
        )


def test_zero_successes_have_lower_bound_zero() -> None:
    assert exact_binomial_lower_confidence_bound(
        event_count=0,
        trial_count=100,
        confidence_level=0.95,
    ) == 0.0


def test_all_successes_match_closed_form_one_sided_clopper_pearson_lower_bound() -> None:
    confidence = 0.95
    trial_count = 100
    bound = exact_binomial_lower_confidence_bound(
        event_count=trial_count,
        trial_count=trial_count,
        confidence_level=confidence,
    )
    expected = (1 - confidence) ** (1 / trial_count)
    assert bound == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_one_success_of_two_lower_bound_is_complement_of_upper_bound() -> None:
    lower = exact_binomial_lower_confidence_bound(
        event_count=1,
        trial_count=2,
        confidence_level=0.95,
    )
    upper = exact_binomial_upper_confidence_bound(
        event_count=1,
        trial_count=2,
        confidence_level=0.95,
    )
    assert lower == pytest.approx(1 - upper, rel=1e-12, abs=1e-15)
    assert lower == pytest.approx(1 - sqrt(0.95), rel=1e-10, abs=1e-12)


def test_lower_bound_is_at_most_observed_rate_and_increases_with_more_successes() -> None:
    few = exact_binomial_lower_confidence_bound(
        event_count=8,
        trial_count=20,
        confidence_level=0.95,
    )
    more = exact_binomial_lower_confidence_bound(
        event_count=16,
        trial_count=20,
        confidence_level=0.95,
    )
    larger_n = exact_binomial_lower_confidence_bound(
        event_count=80,
        trial_count=200,
        confidence_level=0.95,
    )
    same_rate_small = exact_binomial_lower_confidence_bound(
        event_count=8,
        trial_count=20,
        confidence_level=0.95,
    )

    assert few <= 8 / 20
    assert more <= 16 / 20
    assert more > few
    assert larger_n > same_rate_small


def test_lower_bound_rejects_invalid_counts_and_confidence() -> None:
    with pytest.raises(ValueError, match="positive"):
        exact_binomial_lower_confidence_bound(event_count=0, trial_count=0)
    with pytest.raises(ValueError, match="between"):
        exact_binomial_lower_confidence_bound(event_count=3, trial_count=2)
    with pytest.raises(ValueError, match="confidence_level"):
        exact_binomial_lower_confidence_bound(
            event_count=1,
            trial_count=10,
            confidence_level=1.0,
        )

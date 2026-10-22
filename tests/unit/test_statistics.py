"""Tests for statistics helpers."""

from __future__ import annotations

import pytest

from agent_eval.domain.models import TokenUsage
from agent_eval.domain.statistics import latency_stats, mean, percentile, sum_tokens


class TestPercentile:
    def test_single_value(self) -> None:
        assert percentile([5.0], 50) == 5.0

    def test_median_of_odd_count(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_interpolates_even_count(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_p0_is_minimum(self) -> None:
        assert percentile([4.0, 1.0, 3.0], 0) == 1.0

    def test_p100_is_maximum(self) -> None:
        assert percentile([4.0, 1.0, 3.0], 100) == 4.0

    def test_p95(self) -> None:
        assert percentile([float(i) for i in range(1, 101)], 95) == pytest.approx(95.05)

    def test_unsorted_input(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 50) == 2.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            percentile([], 50)

    @pytest.mark.parametrize("q", [-1, 101])
    def test_out_of_range_raises(self, q: int) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0], q)


class TestLatencyStats:
    def test_empty_returns_zeroed(self) -> None:
        stats = latency_stats([])
        assert stats.count == 0
        assert stats.average_ms == 0.0
        assert stats.p50_ms == 0.0

    def test_computes_all_fields(self) -> None:
        stats = latency_stats([100.0, 200.0, 300.0])
        assert stats.count == 3
        assert stats.average_ms == pytest.approx(200.0)
        assert stats.p50_ms == pytest.approx(200.0)
        assert stats.max_ms == 300.0

    def test_average_s_converts_units(self) -> None:
        assert latency_stats([2000.0]).average_s == pytest.approx(2.0)


class TestMean:
    def test_mean(self) -> None:
        assert mean([1.0, 3.0]) == 2.0

    def test_empty_is_none(self) -> None:
        assert mean([]) is None


class TestSumTokens:
    def test_skips_none(self) -> None:
        total = sum_tokens([TokenUsage(input_tokens=5), None, TokenUsage(output_tokens=3)])
        assert total == TokenUsage(input_tokens=5, output_tokens=3)

    def test_empty_list(self) -> None:
        assert sum_tokens([]) == TokenUsage()
        assert sum_tokens([None, None]) == TokenUsage()

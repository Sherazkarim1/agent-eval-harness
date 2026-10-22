# Evaluation report - customer_support@1.0.0

- **Run**: `run_20261006_000636_520c51a1`
- **Agent**: mock-customer-support 1.0.0 (`mock`)
- **Judge**: heuristic-judge via `heuristic-v1`
- **Started**: 2026-10-06T00:06:36.459350+05:00

> **MOCK RUN.** Scores below come from a mock agent and are not real measurements. Use `--mock false` for a live run.

## Summary

| Metric | Value |
| --- | --- |
| Total | 7 |
| Passed | 7 |
| Failed | 0 |
| Errored | 0 |
| Pass rate | 1.0 |
| Overall score | 0.9272 |
| Deterministic | 1.0000 |
| Judge | 0.7283 |
| Trajectory | 1.0000 |
| Performance | 1.0000 |

## Latency

| Percentile | Latency |
| --- | --- |
| Average | 0.111 ms |
| p50 | 0.0948 ms |
| p95 | 0.1791 ms |
| Max | 0.2057 ms |

## Tokens & cost

| Metric | Value |
| --- | --- |
| Input tokens | 840 |
| Output tokens | 400 |
| Total tokens | 1240 |
| API cost | $0.0000 |

_API cost only. Local inference has non-zero compute cost that this tool does not measure._

## Cases

| Case | Status | Score | Latency | Tags |
| --- | --- | --- | --- | --- |
| refund_request_001 | PASS | 0.9305 | 0.2057 ms | billing |
| order_status_001 | PASS | 0.9187 | 0.1171 ms | shipping |
| order_lookup_001 | PASS | 0.9583 | 0.0948 ms | shipping |
| account_balance_001 | PASS | 0.7454 | 0.077 ms | account |
| json_response_001 | PASS | 0.9375 | 0.0961 ms | structured-output |
| tool_failure_001 | PASS | 1.0 | 0.0937 ms | resilience |
| latency_budget_001 | PASS | 1.0 | 0.0926 ms | performance |

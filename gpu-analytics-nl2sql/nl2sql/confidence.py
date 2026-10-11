"""
Confidence from token log-probabilities, and when to ask instead of answer.

An ambiguous question ("top customers last month" - by spend? by visits? which month?)
shows up in the model's own uncertainty: tokens where several continuations were
plausible get low probability. Two summaries are used:

    geometric-mean probability  exp(mean logprob) - overall certainty of the query
    weakest token               exp(min logprob)  - one uncertain choice (a column, a
                                                    filter value) is often what makes a
                                                    query wrong

Tokens outside the SQL (code fences, a leading "Here is the query") are near-certain and
pull the geometric mean up; the weakest-token check is unaffected by them, which is one
reason both are used.

If either falls below its threshold, the pipeline returns a clarifying question rather
than running SQL it is not sure of. Thresholds are calibrated on a labelled question set
(eval/); the defaults here are placeholders, not tuned values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    geo_mean_prob: float
    min_prob: float

    def low(self, mean_threshold: float, min_threshold: float) -> bool:
        return self.geo_mean_prob < mean_threshold or self.min_prob < min_threshold


def score(token_logprobs: list[float]) -> Confidence:
    if not token_logprobs:
        return Confidence(0.0, 0.0)          # no evidence is treated as no confidence
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    return Confidence(math.exp(mean_lp), math.exp(min(token_logprobs)))

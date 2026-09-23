"""Token pricing for cost estimates on the LLM path.

Rates are Anthropic first-party list prices in USD per million tokens.
Amazon Bedrock bills Claude separately (https://aws.amazon.com/bedrock/pricing/),
so treat the result as an estimate; pass --price-in / --price-out to the
runner to use your actual rates. Cache writes are billed at 1.25x input
(5-minute TTL) and cache reads at 0.1x input.
"""

from __future__ import annotations

# Longest key first, so "claude-opus-5-5" isn't read as "claude-opus-5".
LIST_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def rates_for(model_id: str) -> tuple[float, float] | None:
    """(input, output) $/MTok for a Bedrock or first-party model id, or None if unknown.

    Bedrock ids like "us.anthropic.claude-sonnet-4-5-20250929-v1:0" are
    matched on the first-party name they contain.
    """
    for key in sorted(LIST_PRICES, key=len, reverse=True):
        if key in model_id:
            return LIST_PRICES[key]
    return None


def cost_usd(usage: dict, rates: tuple[float, float] | None) -> float | None:
    if rates is None:
        return None
    price_in, price_out = rates
    return (
        usage["input_tokens"] * price_in
        + usage["cache_write_tokens"] * price_in * CACHE_WRITE_MULTIPLIER
        + usage["cache_read_tokens"] * price_in * CACHE_READ_MULTIPLIER
        + usage["output_tokens"] * price_out
    ) / 1_000_000

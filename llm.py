"""Helpers for classifying bookkeeping transactions with OpenAI."""

import os
import json
from openai import OpenAI


# Instantiate the OpenAI client with the user's API key from the environment.
# Reading the key from an environment variable makes the script easier to run
# without modifying the source code.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Some example pairs: (note, bookkeeping category)
EXAMPLES = [
    (
        "Description: Starbuchs LTD 0817. Amount: 6.18. Note: Coffee with client at Starbucks.",
        "Meals & Entertainment",
    ),
    (
        "Description: Regis Congressional Blvd. Amount: 679.00. Note: Office rent for March.",
        "Office Expenses",
    ),
    (
        "Description: Meta Corporation Marketplace. Amount: 21.55. Note: Facebook ad campaign",
        "Advertising",
    ),
    (
        "Description: AT&T Business. Amount: 70.00. Note: Monthly office internet bill.",
        "Utilities",
    ),
    (
        "Description: Uber LTD. Amount: 50.00. Note: Ridde to airport for TikTok training.",
        "Travel",
    ),
    (
        "Description: AMZN Digital. Amount: 16.99. Note: Research and dev book.",
        "Research & Development",
    ),
]


def categorize_expense(description: str, amount: float, note: str) -> str:
    """Return a bookkeeping category for the given transaction.

    Parameters
    ----------
    description:
        Short description of the payee or transaction.
    amount:
        Monetary amount of the transaction.
    note:
        Free-form note provided by the user.
    """

    # We'll build a conversation of role/content objects. "developer" acts like a
    # system-level instruction.
    messages = [
        {
            "role": "developer",
            "content": (
                "You are a helpful bookkeeper that assigns categories to business expenses. "
                "Respond ONLY with the best-fitting category name."
            ),
        }
    ]

    # Provide few-shot examples as user + assistant pairs
    for ex_note, ex_cat in EXAMPLES:
        messages.append({"role": "user", "content": ex_note})
        messages.append({"role": "assistant", "content": ex_cat})

    # Then add the user's transaction
    messages.append(
        {
            "role": "user",
            "content": f"Description: {description}. Amount: {amount}. Note: {note}",
        }
    )

    # Call the 'responses' API. The ``o3`` model is deterministic and does
    # not accept a ``temperature`` parameter.
    response = client.responses.create(model="o3", input=messages)

    # 'response.output_text' should contain the model's final reply
    return response.output_text.strip()


def normalize_payees(payees: list[str]) -> dict[str, str]:
    """Use the LLM to normalize a batch of payee names.

    Parameters
    ----------
    payees:
        List of raw payee strings from bank statements.

    Returns
    -------
    dict
        Mapping of each original payee to a canonical vendor name.
    """

    if not payees:
        return {}

    messages = [
        {
            "role": "developer",
            "content": (
                "You clean merchant names for bookkeeping. Given a list of payees, "
                "return a JSON object mapping each original payee string to a "
                "concise canonical vendor name. Respond ONLY with JSON."
            ),
        },
        {"role": "user", "content": "\n".join(payees)},
    ]

    response = client.responses.create(model="o3", input=messages)
    text = response.output_text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback to identity mapping if parsing fails
        return {p: p for p in payees}


# EXAMPLES = [
#     ("Coffee with client at Starbucks", "Meals & Entertainment"),
#     ("Office rent for March", "Office Expenses"),
#     ("Facebook ad campaign", "Advertising"),
#     ("Monthly internet bill", "Utilities"),
#     ("Uber ride to airport", "Travel"),
# ]

import pandas as pd
from utils import (
    load_existing_table,
    save_table,
    confirm_category,
    save_summary_table,
    propagate_vendor_info,
    normalize_payee,
)
from llm import categorize_expense, normalize_payees as llm_normalize_payees


def main(data, account_type: str | None = None):
    """Categorize new transactions and save them to the output table.

    This function walks through the following flow:

    1. Load existing categorized transactions.
    2. Group transactions by raw payee name and use an LLM to collapse variants
       down to canonical vendors.
    3. For each normalized payee appearing in the new data, further break down
       transactions by significant payment amount clusters and ask the user to
       describe the expense.
    4. Use that description to suggest a bookkeeping category which the user can
       confirm or override.
    5. Record the note and category for all transactions belonging to that
       vendor cluster.
    6. After all transactions have been categorized, aggregate totals by
       category for downstream reporting.

    Parameters
    ----------
    data: DataFrame
        New bank transactions to categorize.
    account_type: str | None, optional
        If provided, read from and write to tables specific to this account
        type (e.g., ``business`` or ``personal``).
    """

    existing = load_existing_table(account_type=account_type)

    # Ensure consistent column types
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Group by raw payee names first, clean them up, then use the LLM to
    # collapse similar payees down to canonical vendors.
    existing["normalized_payee"] = existing["payee"].apply(normalize_payee)
    data["normalized_payee"] = data["payee"].apply(normalize_payee)

    all_payees = pd.concat(
        [existing["normalized_payee"], data["normalized_payee"]]
    ).dropna().unique().tolist()
    mapping = llm_normalize_payees(all_payees)
    existing["normalized_payee"] = existing["normalized_payee"].replace(mapping)
    data["normalized_payee"] = data["normalized_payee"].replace(mapping)

    # Avoid re-processing transactions we have already saved
    existing["transaction_key"] = existing.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    data["transaction_key"] = data.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    processed_keys = set(existing["transaction_key"])
    unprocessed_data = data[~data["transaction_key"].isin(processed_keys)]

    print(f"{len(unprocessed_data)} new transactions need categorization.\n")

    def split_amount_clusters(group: pd.DataFrame, ratio: float = 3.0) -> list[pd.DataFrame]:
        """Break a vendor's transactions into clusters by amount.

        Amounts within ``ratio`` of each other are grouped together. This helps
        surface multiple services from the same vendor and outliers that should
        be categorized separately.
        """

        unique_amounts = sorted(group["amount"].abs().unique())
        if not unique_amounts:
            return [group]
        clusters: list[list[float]] = [[unique_amounts[0]]]
        for amt in unique_amounts[1:]:
            if amt / clusters[-1][0] <= ratio:
                clusters[-1].append(amt)
            else:
                clusters.append([amt])
        return [group[group["amount"].abs().isin(c)] for c in clusters]

    # Prompt once per normalized vendor and amount cluster
    for payee, group in unprocessed_data.groupby("normalized_payee"):
        for sub in split_amount_clusters(group):
            display_payee = payee
            print(f"\nProcessing '{display_payee}' ({len(sub)} transactions)")

            # Show payment statistics before requesting a description
            counts = sub["amount"].value_counts().sort_index()
            print("Payment summary:")
            for amt, count in counts.items():
                print(f"  {count} payment(s) of ${amt:.2f}")

            if account_type == "personal":
                note = input(
                    "Describe the expense or press enter if personal: "
                ).strip()
                if note:
                    category = categorize_expense(
                        display_payee, sub["amount"].mean(), note
                    )
                    category = confirm_category(category)
                else:
                    category = "Personal"
            else:
                note = input("Describe the expense: ").strip()
                category = categorize_expense(
                    display_payee, sub["amount"].mean(), note
                )
                category = confirm_category(category)

            # Apply the note/category to all transactions in the cluster
            sub = sub.assign(note=note, category=category)
            existing = pd.concat([existing, sub], ignore_index=True)
            amount_range = (sub["amount"].min(), sub["amount"].max())
            existing = propagate_vendor_info(
                existing, payee, note, category, amount_range
            )
            save_table(existing, account_type=account_type)
            print(
                f"✅ Saved {len(sub)} transaction(s) for '{display_payee}' as '{category}'"
            )

    # Once everything is categorized, update the summary table
    save_summary_table(existing, account_type=account_type)


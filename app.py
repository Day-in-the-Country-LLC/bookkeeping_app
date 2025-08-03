import pandas as pd
from utils import load_existing_table, save_table, normalize_payee, confirm_category
from llm import categorize_expense


def main(data, account_type="business"):
    """Categorize new transactions and save them to the output table.

    Parameters
    ----------
    data: DataFrame
        New bank transactions to categorize.
    account_type: str
        Either "personal" or "business" to control default assumptions.
    """

    existing = load_existing_table()

    # Ensure consistent column types
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Normalize payee names for smarter grouping
    existing["normalized_payee"] = existing["payee"].apply(normalize_payee)
    data["normalized_payee"] = data["payee"].apply(normalize_payee)

    # Mark each row with a composite key so we don’t re-insert duplicates
    existing["transaction_key"] = existing.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    data["transaction_key"] = data.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )

    processed_keys = set(existing["transaction_key"])
    unprocessed_data = data[~data["transaction_key"].isin(processed_keys)]

    print(f"{len(unprocessed_data)} new transactions need categorization.\n")

    # Group new transactions by normalized payee so we categorize once per vendor
    for payee, group in unprocessed_data.groupby("normalized_payee"):
        payee_history = existing[existing["normalized_payee"] == payee]
        display_payee = group.iloc[0]["payee"]
        used_categories = payee_history["category"].dropna().unique()

        # Determine the default category for this payee based on account type
        if account_type == "personal":
            default_category = "PERSONAL"
            default_note = "Personal expense"
        else:
            if len(used_categories) == 0:
                # Brand new payee, let the model classify
                print(f"🆕 New payee: '{display_payee}'")
                note = input("Describe the expense in plain English: ")
                default_category = categorize_expense(display_payee, group["amount"].mean(), note)
                default_note = note
            elif len(used_categories) == 1:
                auto_cat = used_categories[0]
                print(f"Previously, '{display_payee}' was always categorized as '{auto_cat}'")
                choice = input("[y] Reuse category, [n] new LLM, or [p] personal? ").strip().lower()
                if choice == "y":
                    cat_rows = payee_history[payee_history["category"] == auto_cat]
                    last_note = cat_rows.iloc[-1]["note"]
                    default_category = auto_cat
                    default_note = last_note if pd.notna(last_note) else ""
                elif choice == "p":
                    default_category = "PERSONAL"
                    default_note = "Personal expense"
                else:
                    note = input("Describe the expense in plain English: ")
                    default_category = categorize_expense(display_payee, group["amount"].mean(), note)
                    default_note = note
            else:
                print(f"\nWe’ve seen multiple categories for '{display_payee}' in the past:")
                for i, cat in enumerate(used_categories):
                    print(f"{i}. {cat}")
                choice = input(
                    "Pick a category index, or press [enter] for new LLM, or [p] personal: "
                ).strip().lower()
                if choice.isdigit():
                    idx_choice = int(choice)
                    if 0 <= idx_choice < len(used_categories):
                        selected_cat = used_categories[idx_choice]
                        cat_rows = payee_history[payee_history["category"] == selected_cat]
                        last_note = cat_rows.iloc[-1]["note"]
                        default_category = selected_cat
                        default_note = last_note if pd.notna(last_note) else ""
                    else:
                        note = input("Describe the expense in plain English: ")
                        default_category = categorize_expense(display_payee, group["amount"].mean(), note)
                        default_note = note
                elif choice == "p":
                    default_category = "PERSONAL"
                    default_note = "Personal expense"
                else:
                    note = input("Describe the expense in plain English: ")
                    default_category = categorize_expense(display_payee, group["amount"].mean(), note)
                    default_note = note

        # Combine historical and new amounts to detect outliers
        amounts_history = pd.concat([payee_history["amount"], group["amount"]])
        mean_amount = amounts_history.mean()
        std_amount = amounts_history.std()

        for _, row in group.iterrows():
            amount = row["amount"]
            date = row["date"]

            is_outlier = (
                pd.notna(std_amount)
                and std_amount > 0
                and abs(amount - mean_amount) > 2 * std_amount
            )

            if is_outlier:
                print(
                    f"⚠️ Unusual amount for {display_payee}: ${amount:.2f} (mean ${mean_amount:.2f})"
                )
                note = input(
                    "Describe this transaction, or prefix with 'p ' for personal: "
                )
                if note.startswith("p "):
                    category = "PERSONAL"
                    note = note[2:].strip()
                else:
                    category = categorize_expense(display_payee, amount, note)
            else:
                category = default_category
                note = default_note

            category = confirm_category(category)

            new_row = {
                "payee": row["payee"],
                "normalized_payee": payee,
                "date": date,
                "amount": amount,
                "note": note,
                "category": category,
            }
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            save_table(existing)
            print(
                f"✅ Saved: {row['payee']} [{category}] on {date.date()} => ${amount:.2f}\n"
            )

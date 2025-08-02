import pandas as pd
from utils import load_statements, load_existing_table, save_table
from llm import categorize_expense

def main(data):

    existing = load_existing_table()

    # Make sure both dataframes have the same, consistent column types
    existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Mark each row with a composite key, so we don’t re-insert duplicates
    existing['transaction_key'] = existing.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    data['transaction_key'] = data.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )

    processed_keys = set(existing['transaction_key'])
    unprocessed_data = data[~data['transaction_key'].isin(processed_keys)]

    print(f"{len(unprocessed_data)} new transactions need categorization.\n")

    for idx, row in unprocessed_data.iterrows():
        payee = row['payee']
        date = row['date']
        amount = row['amount']

        # Look up all historical transactions (any date, any amount) for this payee
        payee_history = existing[existing['payee'] == payee]
        used_categories = payee_history['category'].unique()
        used_categories = [cat for cat in used_categories if pd.notna(cat)]

        # 1) If this payee is brand new, do the normal LLM flow
        if len(used_categories) == 0:
            print(f"🆕 New payee: '{payee}'")
            print(f"Transaction date: {date.date()} | Amount: ${amount:.2f}")
            note = input("Describe the expense in plain English - type p+space+note for personal expense: ")
            if note.startswith('p '):
                category = "PERSONAL"
                note = note[2:].strip()
            else:
                category = categorize_expense(note)

        # 2) If no prior category
        if len(used_categories) == 0:
            # Let user quickly mark as personal, or proceed to LLM
            choice = input("Is this a personal expense? (y/n): ").strip().lower()
            if choice == 'y':
                category = "PERSONAL"
                note = "Personal expense"
            else:
                # Normal note + LLM
                note = input("Describe the expense in plain English or type p+space+note for personal expense: ")
                if note.startswith('p '):
                    category = "PERSONAL"
                    note = note[2:].strip()
                else:
                    category = categorize_expense(note)

        # 3) If exactly one prior category
        elif len(used_categories) == 1:
            auto_cat = used_categories[0]
            print(f"Previously, '{payee}' was always categorized as '{auto_cat}'")
            choice = input("[y] Reuse category, [n] new LLM, or [p] personal? ").strip().lower()

            if choice == 'y':
                # Reuse
                cat_rows = payee_history[payee_history['category'] == auto_cat]
                last_note = cat_rows.iloc[-1]['note']
                category = auto_cat
                note = last_note if pd.notna(last_note) else ""
            elif choice == 'p':
                category = "PERSONAL"
                note = "Personal expense"
            else:
                # New LLM
                note = input("Describe the expense in plain English: ")
                category = categorize_expense(note)

        # 4) Multiple prior categories
        else:
            print("\nWe’ve seen multiple categories for this payee in the past:")
            for i, cat in enumerate(used_categories):
                print(f"{i}. {cat}")
            choice = input("Pick a category index, or press [enter] for new LLM, or [p] personal: ").strip().lower()

            if choice.isdigit():
                idx_choice = int(choice)
                if 0 <= idx_choice < len(used_categories):
                    selected_cat = used_categories[idx_choice]
                    cat_rows = payee_history[payee_history['category'] == selected_cat]
                    last_note = cat_rows.iloc[-1]['note']
                    category = selected_cat
                    note = last_note if pd.notna(last_note) else ""
                else:
                    # fallback
                    note = input("Describe the expense in plain English: ")
                    category = categorize_expense(note)
            elif choice == 'p':
                category = "PERSONAL"
                note = "Personal expense"
            else:
                # Do new LLM
                note = input("Describe the expense in plain English: ")
                category = categorize_expense(note)

        # 5) Append the new row to existing and save
        new_row = {
            "payee": payee,
            "date": date,
            "amount": amount,
            "note": note,
            "category": category
        }
        existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        save_table(existing)

        print(f"✅ Saved: {payee} [{category}] on {date.date()} => ${amount:.2f}\n")

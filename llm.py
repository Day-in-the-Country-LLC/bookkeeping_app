from openai import OpenAI

# Instantiate the OpenAI client with your API key
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Some example pairs: (note, bookkeeping category)
EXAMPLES = [
    ("Description: Starbuchs LTD 0817. Amount: 6.18. Note: Coffee with client at Starbucks.", "Meals & Entertainment"),
    ("Description: Regis Congressional Blvd. Amount: 679.00. Note: Office rent for March.", "Office Expenses"),
    ("Description: Meta Corporation Marketplace. Amount: 21.55. Note: Facebook ad campaign", "Advertising"),
    ("Description: AT&T Business. Amount: 70.00. Note: Monthly office internet bill.", "Utilities"),
    ("Description: Uber LTD. Amount: 50.00. Note: Ridde to airport for TikTok training.", "Travel"),
]

def categorize_expense(description: str, amt: str, note: str) -> str:
    """
    Use the new 'responses' style API call to assign a bookkeeping category.
    """

    # We'll build a conversation of role/content objects
    # "developer" seems analogous to a system-level instruction in your snippet
    messages = [{
        "role": "developer",
        "content": (
            "You are a helpful bookkeeper that assigns categories to business expenses. "
            "Respond ONLY with the best-fitting category name."
        )
    }]

    # Provide few-shot examples as user + assistant pairs
    for ex_note, ex_cat in EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Note: {ex_note}"
        })
        messages.append({
            "role": "assistant",
            "content": ex_cat
        })

    # Then your new user query: "Note: <whatever>"
    messages.append({
        "role": "user",
        "content": f"Description: {description}. Amount: {amount}. Note: {note}"
    })

    # Call the new 'responses' API
    response = client.responses.create(
        model="gpt-4o",  # or whichever model name is supported
        input=messages
    )

    # 'response.output_text' should contain the model's final reply
    return response.output_text.strip()


# import openai

# # Replace with your own API key or set this as an environment variable
# openai.api_key = "YOUR_OPENAI_API_KEY"

# EXAMPLES = [
#     ("Coffee with client at Starbucks", "Meals & Entertainment"),
#     ("Office rent for March", "Office Expenses"),
#     ("Facebook ad campaign", "Advertising"),
#     ("Monthly internet bill", "Utilities"),
#     ("Uber ride to airport", "Travel"),
# ]

# def categorize_expense(note):
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful bookkeeper that assigns accounting categories to business expenses."
#         }
#     ]
#     # Provide few-shot examples
#     for ex_note, ex_cat in EXAMPLES:
#         messages.append({"role": "user", "content": f"Note: {ex_note}"})
#         messages.append({"role": "assistant", "content": ex_cat})

#     # Finally, add the user's current request
#     messages.append({"role": "user", "content": f"Note: {note}"})

#     # Call the ChatCompletion API
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0
#     )

#     return response.choices[0].message.content.strip()

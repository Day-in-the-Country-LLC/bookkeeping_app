import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import utils
from llm import categorize_expense, normalize_payees


def test_amzn_digital_normalization():
    payee1 = 'AMZN Digital*GM3C83WE 888-802-3080 WA        09/30'
    payee2 = 'AMZN Digital*K67VZ1R2 888-802-3080 WA        10/30'
    assert utils.normalize_payee(payee1) == 'AMZN DIGITAL'
    assert utils.normalize_payee(payee2) == 'AMZN DIGITAL'


class DummyResponse:
    def __init__(self, text):
        self.output_text = text


def test_llm_called_for_r_and_d(monkeypatch):
    calls = {}

    def fake_create(model, input):
        calls["called"] = True
        return DummyResponse("Research & Development")

    monkeypatch.setattr("llm.client.responses.create", fake_create)

    result = categorize_expense("AMZN Digital", 16.99, "Research & dev")
    assert result == "Research & Development"
    assert calls.get("called")


def test_confirm_category_override(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Software & Subscriptions")
    assert (
        utils.confirm_category("Research & Development")
        == "Software & Subscriptions"
    )

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert utils.confirm_category("Travel") == "Travel"


def test_llm_normalizes_payees(monkeypatch):
    payee1 = 'AMZN Digital*GM3C83WE 888-802-3080 WA        09/30'
    payee2 = 'AMZN Digital*K67VZ1R2 888-802-3080 WA        10/30'

    def fake_create(model, input):
        mapping = {
            payee1: "AMZN DIGITAL",
            payee2: "AMZN DIGITAL",
        }
        return DummyResponse(json.dumps(mapping))

    calls = {}
    def wrapper(model, input):
        calls["called"] = True
        return fake_create(model, input)

    monkeypatch.setattr("llm.client.responses.create", wrapper)

    result = normalize_payees([payee1, payee2])
    assert result[payee1] == "AMZN DIGITAL"
    assert result[payee2] == "AMZN DIGITAL"
    assert calls.get("called")


def test_propagate_vendor_info():
    df = pd.DataFrame(
        {
            "payee": ["A", "A", "B"],
            "normalized_payee": ["A", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "amount": [1, 2, 3],
            "note": ["", "", ""],
            "category": ["", "", ""],
        }
    )
    updated = utils.propagate_vendor_info(df, "A", "Coffee", "Meals")
    assert (updated.loc[updated["normalized_payee"] == "A", "note"] == "Coffee").all()
    assert (updated.loc[updated["normalized_payee"] == "A", "category"] == "Meals").all()
    assert updated.loc[updated["normalized_payee"] == "B", "note"].iloc[0] == ""


def test_generate_summary():
    df = pd.DataFrame(
        {
            "category": ["Meals", "Meals", "Office"],
            "amount": [10, 20, 30],
        }
    )
    summary = utils.generate_summary(df)
    meals_total = summary.loc[summary["category"] == "Meals", "total_amount"].iloc[0]
    office_total = summary.loc[summary["category"] == "Office", "total_amount"].iloc[0]
    assert meals_total == 30
    assert office_total == 30


def test_account_type_separates_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "llm_normalize_payees", lambda payees: {p: p for p in payees})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    df_business = pd.DataFrame(
        {
            "payee": ["BizCo"],
            "date": pd.to_datetime(["2024-01-01"]),
            "amount": [100],
            "note": [""],
            "category": [""],
        }
    )
    df_personal = pd.DataFrame(
        {
            "payee": ["Home"],
            "date": pd.to_datetime(["2024-02-01"]),
            "amount": [200],
            "note": [""],
            "category": [""],
        }
    )

    utils.save_table(df_business, account_type="business")
    utils.save_table(df_personal, account_type="personal")

    loaded_business = utils.load_existing_table(account_type="business")
    loaded_personal = utils.load_existing_table(account_type="personal")

    assert loaded_business["payee"].tolist() == ["BizCo"]
    assert loaded_personal["payee"].tolist() == ["Home"]

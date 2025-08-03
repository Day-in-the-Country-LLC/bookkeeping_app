import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
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


def test_godaddy_normalization():
    payee1 = 'DNH*GODADDY #1234567 480-5058877 AZ        09/30'
    payee2 = 'DNH*GODADDY #7654321 800-9876543 AZ        10/30'
    assert utils.normalize_payee(payee1) == utils.normalize_payee(payee2) == 'DNH*GODADDY AZ'


def test_adobe_phone_normalization():
    payee = 'ADOBE  *800-833-6687 800-833-6687 CA         02/10'
    assert utils.normalize_payee(payee) == 'ADOBE'


class DummyResponse:
    def __init__(self, text):
        self.output_text = text


def test_llm_called_for_r_and_d(monkeypatch):
    calls = {}

    def fake_create(model, input, **kwargs):
        assert model == "o3"
        assert "temperature" not in kwargs

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

    calls = {}

    def fake_create(model, input, **kwargs):
        assert model == "o3"

        mapping = {
            payee1: "AMZN DIGITAL",
            payee2: "AMZN DIGITAL",
        }
        calls["called"] = True
        return DummyResponse(json.dumps(mapping))

    monkeypatch.setattr("llm.client.responses.create", fake_create)

    result = normalize_payees([payee1, payee2])
    assert result[payee1] == "AMZN DIGITAL"
    assert result[payee2] == "AMZN DIGITAL"
    assert calls.get("called")


def test_categorize_expense_deterministic(monkeypatch):
    def fake_create(model, input, **kwargs):
        assert model == "o3"
        assert "temperature" not in kwargs
        return DummyResponse("Meals & Entertainment")

    monkeypatch.setattr("llm.client.responses.create", fake_create)

    first = categorize_expense("Starbuchs LTD", 6.18, "Coffee with client")
    second = categorize_expense("Starbuchs LTD", 6.18, "Coffee with client")

    assert first == "Meals & Entertainment"
    assert first == second


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
    updated = utils.propagate_vendor_info(
        df, "A", "Coffee", "Meals", amount_range=(0, 2)
    )
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

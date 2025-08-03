import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
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

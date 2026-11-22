"""Request-path tests for the official adapter. No live or paid calls."""


def test_official_provider_posts_anthropic_messages_schema(monkeypatch):
    from piia.intelligence.agents import providers, workflow

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": '{"summary":"anthropic-fixture"}'}],
                    "usage": {"input_tokens": 4, "output_tokens": 1}}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return Response()

    monkeypatch.setattr(workflow, "model_configured", lambda: True)
    monkeypatch.setattr(providers, "model_configured", lambda: True)
    monkeypatch.setattr(workflow, "model_name", lambda: "claude-demo")
    monkeypatch.setattr(providers, "model_name", lambda: "claude-demo")
    monkeypatch.setattr(providers, "model_base_url", lambda: "https://api.anthropic.com")
    monkeypatch.setattr(providers, "model_provider", lambda: "anthropic")
    monkeypatch.setattr(providers.httpx, "Client", Client)
    result = workflow.run_workflow(mode="model", payload={"statement": "x", "evidence": []},
                                   legal_evidence_ids=[])
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "x-api-key" in captured["headers"]
    assert "messages" in captured["body"]
    assert result.researcher.summary == "anthropic-fixture"
    assert result.usage_records[0]["input_tokens"] == 4
    assert result.independent_llm_review == "NOT_WIRED"

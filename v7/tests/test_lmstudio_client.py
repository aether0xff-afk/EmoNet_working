from __future__ import annotations

from types import SimpleNamespace

from emonet_v7.lmstudio_client import LMStudioClient


class FakeModelsAPI:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="local-chat-model"),
                SimpleNamespace(id="local-embedding-model"),
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.models = FakeModelsAPI()


def test_list_models_uses_local_client_boundary() -> None:
    client = LMStudioClient(base_url="http://localhost:1234", model="placeholder")
    client._client = FakeOpenAIClient()
    assert client.base_url == "http://localhost:1234/v1"
    assert client.list_models() == ["local-chat-model", "local-embedding-model"]

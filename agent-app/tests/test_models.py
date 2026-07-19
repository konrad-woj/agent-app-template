import pytest
from pydantic import ValidationError

from app.models import ChatRequest, ReplayRequest


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(thread_id="t-1", message="Research quantum computing")
        assert req.thread_id == "t-1"
        assert req.message == "Research quantum computing"

    def test_blank_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="t-1", message="   ")

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="t-1", message="")

    def test_missing_thread_id_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Research something")  # type: ignore[call-arg]

    def test_blank_thread_id_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="   ", message="Research something")

    def test_missing_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="t-1")  # type: ignore[call-arg]


class TestReplayRequest:
    def test_valid_replay_request(self):
        req = ReplayRequest(checkpoint_id="cp-abc-123")
        assert req.checkpoint_id == "cp-abc-123"

    def test_missing_checkpoint_id_rejected(self):
        with pytest.raises(ValidationError):
            ReplayRequest()  # type: ignore[call-arg]

    def test_blank_checkpoint_id_rejected(self):
        with pytest.raises(ValidationError):
            ReplayRequest(checkpoint_id="   ")

    def test_empty_checkpoint_id_rejected(self):
        with pytest.raises(ValidationError):
            ReplayRequest(checkpoint_id="")

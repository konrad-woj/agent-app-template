from pydantic import BaseModel, Field, ValidationInfo, field_validator

MAX_REQUEST_MESSAGE_LEN = 4096


def _not_blank(value: str, info: ValidationInfo) -> str:
    if not value.strip():
        raise ValueError(f"{info.field_name} must not be blank")
    return value


class ChatRequest(BaseModel):
    thread_id: str = Field(max_length=128)
    message: str = Field(max_length=MAX_REQUEST_MESSAGE_LEN)
    approve: bool | None = None  # None = new turn; True/False = resume interrupt

    _validate_not_blank = field_validator("thread_id", "message")(_not_blank)


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    is_interrupted: bool = False
    interrupt_value: dict | None = None
    final_answer: str | None = None
    guard_reason: str | None = None
    dead_letter: dict | None = None


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    status: str | None
    messages_count: int


class ReplayRequest(BaseModel):
    checkpoint_id: str

    _validate_not_blank = field_validator("checkpoint_id")(_not_blank)

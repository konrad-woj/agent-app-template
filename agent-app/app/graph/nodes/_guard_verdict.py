from pydantic import BaseModel, field_validator


class GuardVerdict(BaseModel):
    verdict: str
    reason: str

    @field_validator("verdict")
    @classmethod
    def normalise_verdict(cls, v: str) -> str:
        return v.strip().lower()

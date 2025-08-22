from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)


def validate_schema(data: dict, schema: Type[T]) -> T:
    """Validate data against a Pydantic schema."""
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")


def validate_request_data(data: dict) -> dict:
    """Validate and sanitize request data."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = value.strip()
        else:
            sanitized[key] = value
    
    return sanitized
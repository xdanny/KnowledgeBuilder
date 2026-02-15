from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError


class StateStore(ABC):
    @abstractmethod
    def read(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def write(self, key: str, value: str) -> None:
        raise NotImplementedError


class S3StateStore(StateStore):
    def __init__(self, s3_client: Any, bucket: str) -> None:
        if not bucket:
            raise ValueError("S3 state bucket is required.")
        self.s3_client = s3_client
        self.bucket = bucket

    def read(self, key: str) -> str | None:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read().decode("utf-8").strip() or None
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404"}:
                return None
            raise

    def write(self, key: str, value: str) -> None:
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=value.encode("utf-8"))


class LocalFileStateStore(StateStore):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def read(self, key: str) -> str | None:
        return self._read_all().get(key)

    def write(self, key: str, value: str) -> None:
        data = self._read_all()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

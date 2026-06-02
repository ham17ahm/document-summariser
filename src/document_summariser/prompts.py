from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


class PromptRenderError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    path: Path
    text: str
    sha256: str

    def render(self, **variables: str) -> str:
        required = set(_PLACEHOLDER_PATTERN.findall(self.text))
        missing = sorted(required - set(variables))
        if missing:
            raise PromptRenderError(f"Missing prompt variables for {self.path}: {', '.join(missing)}")

        def replace(match: re.Match[str]) -> str:
            return variables[match.group(1)]

        return _PLACEHOLDER_PATTERN.sub(replace, self.text)


def load_prompt(path: Path) -> PromptTemplate:
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(path=path, text=text, sha256=digest)

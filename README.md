# headroom
 
**headroom** is a Python library for building and managing LLM prompts within a token budget. Assemble prompts from typed, prioritized fragments — and automatically compact or drop content when you're running low on tokens.

## Concepts
 
### Slots and Importance
 
Fragments are placed into named **slots** that define the structure of your prompt:
 
| Slot | Default Importance | Purpose |
|---|---|---|
| `system` | `CRITICAL` | Model persona and top-level constraints |
| `instructions` | `HIGH` | Task-specific directions |
| `context` | `NORMAL` | Supporting documents, code, data |
| `history` | `LOW` | Prior conversation turns |
| `user` | `CRITICAL` | The current user message |
 
Each fragment is also tagged with an **importance level** (`LOW`, `NORMAL`, `HIGH`, `CRITICAL`). Importance governs which fragments are targeted first during compaction — lower importance fragments are compacted before higher ones, and `CRITICAL` fragments are never dropped.
 
### The `Promptable` Protocol
 
Any object that knows how to render itself into a string implements `Promptable`:
 
```python
from typing import Protocol, runtime_checkable
 
@runtime_checkable
class Promptable(Protocol):
    def to_prompt(self) -> str:
        ...
```
 
Pass a `Promptable` anywhere headroom accepts content and it will call `to_prompt()` at render time. Plain strings are also accepted directly.
 
**Example:**
 
```python
@dataclass
class CodeFragment:
    path: str
    source: str
 
    def to_prompt(self) -> str:
        return f"# {self.path}\n```python\n{self.source}\n```"
 
builder.context(CodeFragment(path="auth/login.py", source=src))
```

### The `Compactable` Protocol

When the rendered prompt exceeds the token budget, headroom applies **compactors** in the order you supply them.  These compactors are applied in an exhaustive manor, meaning they will compact one fragment at a time iteratively, and compact as many fragments as they can before moving onto the next compactor.
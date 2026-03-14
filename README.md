# headroom
 
**headroom** is a Python library for building and managing LLM prompts within a token budget. Assemble prompts from typed, prioritized fragments — and automatically compact or drop content when you're running low on tokens.

## Quick Start
 
```python
from headroom.builder import PromptBuilder, Importance, ExhaustionPolicy
from headroom.builder import InlineCompactor, TruncateCompactor, DropFragCompactor
 
builder = PromptBuilder(
    max_tokens=2000,
    compactors=(InlineCompactor(), TruncateCompactor(max_chars=300), DropFragCompactor()),
    exhaustion_policy=ExhaustionPolicy.WARN,
)
 
builder \
    .system("You are a security auditing assistant.") \
    .instructions("Trace the call chain from the entry point to the database layer.") \
    .context(login_fragment, importance=Importance.HIGH) \
    .context(db_fragment, importance=Importance.NORMAL) \
    .history(previous_exchange, importance=Importance.LOW) \
    .user("Which functions handle user input without sanitization?")
 
result = builder.build()
print(result.prompt)
print(f"Used {result.tokens_used} / {result.token_budget} tokens")
```

`build()` renders all fragments into a prompt string. If the token budget is exceeded, headroom walks slots in compaction order (`history → context → instructions`), applying the compactor stack to fragments from least to most important until the prompt fits.

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

Plain strings are also accepted anywhere `Promptable` is expected.
 
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

Objects that know how to shrink themselves implement `Compactable`:
 
```python
@runtime_checkable
class Compactable(Protocol):
    def compact(self) -> Promptable: ...
```
 
`compact()` returns a smaller `Promptable` representation of the object. The `InlineCompactor` calls this automatically during budget enforcement — implement `Compactable` on your domain objects to control exactly how they degrade.
 
**Example:**
 
```python
@dataclass
class CodeFragment:
    path: str
    source: str
 
    def to_prompt(self) -> str:
        return f"# {self.path}\n```python\n{self.source}\n```"
 
    def compact(self) -> Promptable:
        signatures = extract_signatures(self.source)
        return CodeFragment(path=self.path, source=signatures)
```
 
### Compactors
 
Compactors implement a single method:
 
```python
class Compactor(Protocol):
    def apply(self, fragment: Fragment) -> Fragment | Literal["drop"] | None: ...
```
 
Returning a `Fragment` replaces the original, `"drop"` removes it, and `None` means the compactor has nothing to do with that fragment. The framework owns traversal — compactor authors only decide what to do with a single fragment.
 
**Built-in compactors:**
 
| Compactor | Behavior |
|---|---|
| `InlineCompactor` | Calls `compact()` on fragments whose content implements `Compactable` |
| `TruncateCompactor(max_chars)` | Truncates string fragment content to `max_chars` |
| `DropFragCompactor` | Removes the fragment entirely — use as a last resort |
 
Compactors are tried in the order supplied, exhausted one at a time before moving to the next.
 
### `BuildResult`
 
`build()` returns a `BuildResult` rather than a plain string:
 
```python
result = builder.build()
result.prompt             # the final rendered prompt string
result.tokens_used        # token count of the final prompt
result.token_budget       # the configured max_tokens
result.compaction_events  # tuple of CompactionEvent, one per compaction step taken
```
 
### Exhaustion Policy
 
Control what happens when all compactors are exhausted but the prompt is still over budget:
 
```python
# Log a warning (default)
PromptBuilder(exhaustion_policy=ExhaustionPolicy.WARN)
 
# Raise a ValueError instead
PromptBuilder(exhaustion_policy=ExhaustionPolicy.RAISE)
```
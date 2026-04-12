"""
Session Memory module.

Maintains persistent in-session history of commands and actions
to give the LLM context for follow-up commands.
"""

from dataclasses import dataclass, field
from typing import List
import datetime


@dataclass
class MemoryEntry:
    timestamp: str
    command: str
    action: str
    status: str


class SessionMemory:
    def __init__(self, max_entries: int = 10):
        self.entries: List[MemoryEntry] = []
        self.max_entries = max_entries

    def add(self, command: str, result: dict):
        entry = MemoryEntry(
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
            command=command,
            action=result.get("action_taken", ""),
            status=result.get("status", "unknown")
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_context(self) -> str:
        if not self.entries:
            return "No previous actions in this session."
        lines = []
        for e in self.entries[-5:]:  # last 5 only
            lines.append(f"[{e.timestamp}] Command: '{e.command}' → {e.action} ({e.status})")
        return "\n".join(lines)

    def clear(self):
        self.entries.clear()

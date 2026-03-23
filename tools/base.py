from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    @abstractmethod
    def run(self) -> list:
        """Execute the tool and return a list of results."""
        raise NotImplementedError

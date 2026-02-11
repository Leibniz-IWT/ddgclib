"""
Shared method registry for pluggable computational methods.

Usage
-----
    curvature_methods = MethodRegistry("curvature_i")
    curvature_methods.register("laplace-beltrami", compute_laplace_beltrami)
    fn = curvature_methods["laplace-beltrami"]
    curvature_methods.available()  # ["laplace-beltrami"]
"""

from typing import Callable


class MethodRegistry:
    """Registry for pluggable computational methods.

    Parameters
    ----------
    name : str
        Human-readable name for error messages (e.g., "curvature_i").
    """

    def __init__(self, name: str):
        self.name = name
        self._methods: dict[str, Callable] = {}

    def register(self, key: str, fn: Callable) -> None:
        """Register a method under the given key."""
        self._methods[key] = fn

    def __getitem__(self, key: str) -> Callable:
        if key not in self._methods:
            raise KeyError(
                f"Unknown {self.name} method: {key!r}. "
                f"Available: {list(self._methods.keys())}"
            )
        return self._methods[key]

    def __contains__(self, key: str) -> bool:
        return key in self._methods

    def available(self) -> list[str]:
        """Return list of registered method names."""
        return list(self._methods.keys())

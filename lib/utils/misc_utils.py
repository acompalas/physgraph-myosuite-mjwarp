# ---
# Copied verbatim from PhysGraph (github.com/acompalas/PhysGraph),
# lib/utils/. Confirmed IsaacGym-free (traced 2026-09-05) -- reused
# as-is per the project's inherit-where-possible strategy.
# ---
from typing import List, Union, Callable, Literal

import fnmatch


def _match_patterns_helper(element, patterns):
    for p in patterns:
        if callable(p) and p(element):
            return True
        if fnmatch.fnmatch(element, p):
            return True
    return False


def match_patterns(
    item: str,
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence: Literal["include", "exclude"] = "exclude",
):
    """
    Args:
        include: None to disable `include` filter and delegate to exclude
        precedence: "include" or "exclude"
    """
    assert precedence in ["include", "exclude"]
    if exclude is None:
        exclude = []
    if isinstance(exclude, (str, Callable)):
        exclude = [exclude]
    if isinstance(include, (str, Callable)):
        include = [include]
    if include is None:
        # exclude is the sole veto vote
        return not _match_patterns_helper(item, exclude)

    if precedence == "include":
        return _match_patterns_helper(item, include)
    else:
        if _match_patterns_helper(item, exclude):
            return False
        else:
            return _match_patterns_helper(item, include)

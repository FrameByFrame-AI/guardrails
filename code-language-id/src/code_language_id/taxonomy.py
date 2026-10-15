"""Unified language taxonomy for the 113-class multi-label model.

Background
----------
Two label spaces existed in this project:

* the **lite single-label** model's 107 ``canonical_language`` values (programming
  languages from The Stack + Rosetta Code), and
* the **Accuknox 25-label** multi-label schema, whose output JSON is the format
  we must reproduce.

All 25 Accuknox languages are already inside the source corpus. Nineteen of them
coincide with a lite canonical name; the remaining six config/markup formats
(``Bash, Dockerfile, Makefile, SQL, Terraform, YAML``) have abundant data in
``raw/the-stack-v1`` (3000 rows each) but were *excluded* from the lite
taxonomy. Re-including them yields the unified universe:

    107 (lite canonical)  +  6 (config formats)  =  113 languages.

Canonical-name policy
---------------------
* For the 25 Accuknox languages we use the **Accuknox spelling** (``AWK``,
  ``Batch``, ``C#`` …) because that is the output schema the model must emit.
  This renames two lite values: ``Awk -> AWK`` and ``Batchfile -> Batch``.
* For the other 88 languages we keep the lite ``canonical_language`` spelling.

Ordering
--------
``LANGUAGES`` lists the 25 Accuknox languages first (in their established order),
then the 88 lite-only languages alphabetically. Keeping the 25 in their original
positions lets a new head warm-start from the v7 25-label checkpoint by index,
while ``resolve()`` lets any head warm-start from the lite checkpoint by name.

``resolve(name)`` maps *any* label spelling seen across the data sources — raw
Stack parquet slugs, Rosetta ``raw_language`` variants, lite canonical names,
Accuknox names — onto a canonical entry. Unknown labels raise ``KeyError`` so
that silent label loss can never happen; ``validate_sources()`` asserts coverage.
"""

from __future__ import annotations

import re

# --- The 25 Accuknox languages, in their canonical output order --------------
ACCUKNOX_25: tuple[str, ...] = (
    "AWK", "Bash", "Batch", "C", "C#", "C++", "Dockerfile", "Go",
    "Java", "JavaScript", "Kotlin", "Lua", "Makefile", "Perl",
    "PowerShell", "Python", "R", "Ruby", "Rust", "SQL", "Scala",
    "Swift", "Terraform", "YAML", "jq",
)

# --- The 88 languages that exist only in the lite taxonomy --------------------
# (lite's 107 canonical_language values minus the 19 that map onto ACCUKNOX_25)
LITE_ONLY_88: tuple[str, ...] = (
    "ABAP", "APL", "ARM Assembly", "ATS", "ActionScript", "Ada",
    "AppleScript", "AutoHotkey", "AutoIt", "BASIC", "BQN", "Befunge",
    "COBOL", "Ceylon", "Clojure", "CoffeeScript", "ColdFusion",
    "Common Lisp", "Component Pascal", "Crystal", "D", "Dart", "E",
    "Eiffel", "Elixir", "Emacs Lisp", "Erlang", "Euphoria", "F#",
    "Factor", "Fantom", "Forth", "Fortran", "FreeBASIC", "GAP",
    "Groovy", "Haskell", "Haxe", "IDL", "Io", "J", "Julia", "LFE",
    "LabVIEW", "Lasso", "Logtalk", "M", "M4", "MATLAB", "MAXScript",
    "Mathematica/Wolfram Language", "Mercury", "Modula-2", "Modula-3",
    "Nemerle", "NewLisp", "Nim", "OCaml", "Objective-C", "Oz", "PHP",
    "Pascal", "PicoLisp", "Pike", "Processing", "Prolog", "PureBasic",
    "QuickBASIC", "REXX", "Racket", "Raku", "Rebol", "Red", "Ring",
    "SAS", "Scheme", "Scilab", "Smalltalk", "Standard ML", "Stata",
    "Tcl", "V", "VBA", "VBScript", "Vala", "Visual Basic .NET", "Wren",
    "Zig",
)

LANGUAGES: tuple[str, ...] = ACCUKNOX_25 + LITE_ONLY_88
LANG_TO_IDX: dict[str, int] = {lang: i for i, lang in enumerate(LANGUAGES)}
NUM_LANGUAGES = len(LANGUAGES)
_CANON_SET = set(LANGUAGES)

# --- Raw The-Stack-v1 parquet slug -> canonical ------------------------------
# Slugs that don't title-case cleanly are listed; the rest fall through the
# normalization step in resolve(). All 92 stack slugs are covered explicitly to
# keep the mapping reviewable.
STACK_SLUG_ALIASES: dict[str, str] = {
    "abap": "ABAP", "actionscript": "ActionScript", "ada": "Ada",
    "apl": "APL", "applescript": "AppleScript", "ats": "ATS",
    "autohotkey": "AutoHotkey", "autoit": "AutoIt", "awk": "AWK",
    "bash": "Bash", "batchfile": "Batch", "befunge": "Befunge", "c": "C",
    "c_plus_plus": "C++", "c_sharp": "C#", "ceylon": "Ceylon",
    "clojure": "Clojure", "cobol": "COBOL", "coffeescript": "CoffeeScript",
    "coldfusion": "ColdFusion", "common_lisp": "Common Lisp",
    "component_pascal": "Component Pascal", "crystal": "Crystal", "d": "D",
    "dart": "Dart", "dockerfile": "Dockerfile", "eiffel": "Eiffel",
    "elixir": "Elixir", "emacs_lisp": "Emacs Lisp", "erlang": "Erlang",
    "f": "F#", "factor": "Factor", "fantom": "Fantom", "forth": "Forth",
    "fortran": "Fortran", "gap": "GAP", "go": "Go", "groovy": "Groovy",
    "haskell": "Haskell", "haxe": "Haxe", "idl": "IDL", "io": "Io",
    "j": "J", "java": "Java", "javascript": "JavaScript", "julia": "Julia",
    "kotlin": "Kotlin", "labview": "LabVIEW", "lasso": "Lasso", "lfe": "LFE",
    "logtalk": "Logtalk", "lua": "Lua", "m": "M", "m4": "M4",
    "makefile": "Makefile",
    "mathematica_wolfram_language": "Mathematica/Wolfram Language",
    "matlab": "MATLAB", "maxscript": "MAXScript", "nim": "Nim",
    "ocaml": "OCaml", "oz": "Oz", "pascal": "Pascal", "perl": "Perl",
    "php": "PHP", "pike": "Pike", "powershell": "PowerShell",
    "processing": "Processing", "prolog": "Prolog", "purebasic": "PureBasic",
    "python": "Python", "r": "R", "racket": "Racket", "raku": "Raku",
    "rebol": "Rebol", "red": "Red", "ruby": "Ruby", "rust": "Rust",
    "sas": "SAS", "scala": "Scala", "scheme": "Scheme", "scilab": "Scilab",
    "smalltalk": "Smalltalk", "sql": "SQL", "standard_ml": "Standard ML",
    "stata": "Stata", "swift": "Swift", "tcl": "Tcl", "terraform": "Terraform",
    "vala": "Vala", "visual_basic_net": "Visual Basic .NET", "yaml": "YAML",
    "zig": "Zig",
}

# --- Structural variants (Rosetta raw_language, lite names, etc.) -------------
# Only entries the normalization step can't resolve on its own are listed.
EXPLICIT_ALIASES: dict[str, str] = {
    # lite canonical names renamed to the Accuknox spelling
    "Awk": "AWK",
    "Batchfile": "Batch",
    # Rosetta / linguist raw_language spellings that aren't mere case variants
    "Delphi": "Pascal",
    "Object Pascal": "Pascal",
    "Perl6": "Raku",
    "Nimrod": "Nim",
    "Octave": "MATLAB",
    "OCTAVE": "MATLAB",
    "MUMPS": "M",
    "Vlang": "V",
    "Visual Basic": "Visual Basic .NET",
    "Go!": "Go",
    "Lisp": "Common Lisp",
    "QB64": "QuickBASIC",
    "QBasic": "QuickBASIC",
    "Quick BASIC": "QuickBASIC",
    "WinBatch": "Batch",
    "Batch File": "Batch",
    "Mathematica": "Mathematica/Wolfram Language",
    "Wolfram Language": "Mathematica/Wolfram Language",
    "Wolfram Language/Mathematica": "Mathematica/Wolfram Language",
    "Mathematica / Wolfram Language": "Mathematica/Wolfram Language",
    "Mathematica/ Wolfram Language": "Mathematica/Wolfram Language",
    "Mathematica//Wolfram Language": "Mathematica/Wolfram Language",
}


def _norm(s: str) -> str:
    """Case/space-insensitive key that preserves distinguishing symbols.

    Keeps ``+ # . /`` so that ``C``, ``C#`` and ``C++`` stay distinct; collapses
    internal whitespace and lowercases everything else.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# normalized-form lookup, built from canonical names + all known aliases
_NORM_TO_CANON: dict[str, str] = {}
for _canon in LANGUAGES:
    _NORM_TO_CANON[_norm(_canon)] = _canon
for _src, _dst in {**STACK_SLUG_ALIASES, **EXPLICIT_ALIASES}.items():
    _NORM_TO_CANON.setdefault(_norm(_src), _dst)


def resolve(name: str, *, strict: bool = True) -> str | None:
    """Map any source label spelling to a canonical ``LANGUAGES`` entry.

    Raises ``KeyError`` on an unknown label when ``strict`` (the default), else
    returns ``None`` — so callers can choose between failing loudly during data
    construction and skipping unmapped rows.
    """
    if name is None:
        if strict:
            raise KeyError("cannot resolve None language label")
        return None
    s = str(name).strip()
    if s in _CANON_SET:
        return s
    if s in EXPLICIT_ALIASES:
        return EXPLICIT_ALIASES[s]
    if s in STACK_SLUG_ALIASES:
        return STACK_SLUG_ALIASES[s]
    hit = _NORM_TO_CANON.get(_norm(s))
    if hit is not None:
        return hit
    if strict:
        raise KeyError(f"unknown language label {name!r}")
    return None


def stack_slug_to_canonical(slug: str) -> str:
    """Resolve a ``raw/the-stack-v1`` parquet stem to its canonical language."""
    return resolve(slug)


# sanity checks that run at import time — cheap and catch taxonomy drift early
assert len(LANGUAGES) == 113, f"expected 113 languages, got {len(LANGUAGES)}"
assert len(_CANON_SET) == len(LANGUAGES), "duplicate canonical language name"
assert NUM_LANGUAGES == 113

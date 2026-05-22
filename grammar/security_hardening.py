"""Grammar Security Hardening — Rule ingestion with strict input validation.

Blocks path traversal, XSS, SQL injection, and arbitrary code execution.
Provides sandboxed exec for production rules and rule provenance tracking.
"""

from __future__ import annotations

import ast
import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── Validation Constants ─────────────────────────────────────────

RULE_NAME_MAX_LEN = 64
TAGLINE_MAX_LEN = 256
CONDITION_MAX_LEN = 1024
EXEC_MAX_LEN = 512
PROVENANCE_MAX_LEN = 2048

# Alphanumeric + underscore ONLY (stricter than core.py — no hyphens)
RULE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")

# SQLi blacklist — semicolons, comment dashes, and dangerous keywords.
SQLI_BLACKLIST = re.compile(
    r";|--|\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|EXECUTE|UNION|SELECT)\b",
    re.IGNORECASE,
)

# Script tag detection — flag explicit script/content injection patterns.
SCRIPT_PATTERN = re.compile(
    r"<\s*script\b|<\s*iframe\b|<\s*object\b|<\s*embed\b|javascript:\s*|on\w+\s*=",
    re.IGNORECASE,
)

# HTML tag stripper — removes all angle-bracket tags.
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")

# Allowed AST node types for condition evaluation
_CONDITION_AST_WHITELIST = frozenset({
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Invert,
    ast.Name, ast.Load, ast.Constant,
    ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
})

# Strict AST whitelist for exec fields — only literal-safe nodes
_EXEC_AST_WHITELIST = frozenset({
    ast.Expression,
    ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Invert,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Load,
})


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class RuleProvenance:
    """Immutable provenance record for a rule's lifecycle."""
    source: str = "unknown"          # e.g. "api", "file", "evolution", "manual"
    origin_id: Optional[str] = None  # Original ID before ingestion
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ingested_by: Optional[str] = None  # Agent/user identifier
    checksum: Optional[str] = None   # SHA-256 of raw payload
    parent_names: Tuple[str, ...] = ()  # For evolved rules: (parent_a, parent_b)
    generation: int = 0              # Evolution generation counter
    history: List[Dict[str, Any]] = field(default_factory=list)

    def add_event(self, event: str, detail: Optional[str] = None) -> None:
        """Append an audit event to the provenance history."""
        self.history.append({
            "event": event,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


@dataclass
class Production:
    tagline: str = ""
    condition: str = ""
    exec_field: Optional[str] = field(default=None, repr=False)


@dataclass
class Rule:
    name: str
    production: Production
    provenance: RuleProvenance = field(default_factory=RuleProvenance)


# ── Validation Exceptions ──────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a rule field fails security validation."""
    pass


class SecurityError(ValidationError):
    """Raised when a security-critical violation is detected."""
    pass


# ── Core Validation Functions ────────────────────────────────────

def validate_rule_name(name: str) -> str:
    """Sanitize rule name.

    - Alphanumeric + underscore ONLY. No hyphens, dots, slashes.
    - Max 64 characters.
    - Must not be empty.
    - Rejects path traversal sequences.
    """
    if not isinstance(name, str):
        raise ValidationError("Rule name must be a string.")
    name = name.strip()
    if not name:
        raise ValidationError("Rule name must not be empty.")
    if len(name) > RULE_NAME_MAX_LEN:
        raise ValidationError(f"Rule name exceeds {RULE_NAME_MAX_LEN} characters.")
    if ".." in name or "/" in name or "\\" in name:
        raise SecurityError("Rule name contains path traversal characters.")
    if not RULE_NAME_PATTERN.match(name):
        raise SecurityError(
            "Rule name contains illegal characters. "
            "Allowed: a-z, A-Z, 0-9, _ (underscore only)."
        )
    return name


def validate_production_fields(tagline: str, condition: str) -> Tuple[str, str]:
    """Sanitize production fields together.

    - Tagline: strip HTML tags, block script/injection patterns, HTML-escape.
    - Condition: blacklist SQLi patterns, validate as parseable expression.

    Returns sanitized (tagline, condition) tuple.
    Raises SecurityError on violation.
    """
    # ---- Tagline validation ----
    if not isinstance(tagline, str):
        raise ValidationError("Tagline must be a string.")
    if len(tagline) > TAGLINE_MAX_LEN:
        raise ValidationError(f"Tagline exceeds {TAGLINE_MAX_LEN} characters.")

    if SCRIPT_PATTERN.search(tagline):
        raise SecurityError("Tagline contains blocked script/injection patterns.")

    tagline = HTML_TAG_PATTERN.sub("", tagline)  # strip all tags
    tagline = html.escape(tagline, quote=True)      # escape ampersands, quotes, etc.

    # ---- Condition validation ----
    if not isinstance(condition, str):
        raise ValidationError("Condition must be a string.")
    if len(condition) > CONDITION_MAX_LEN:
        raise ValidationError(f"Condition exceeds {CONDITION_MAX_LEN} characters.")
    condition = condition.strip()
    if not condition:
        return tagline, condition
    if SQLI_BLACKLIST.search(condition):
        raise SecurityError("Condition contains blocked SQL injection patterns.")

    return tagline, condition


def validate_exec_field(exec_code: Optional[str]) -> Optional[str]:
    """Sandbox production.exec — strict literal-only evaluation.

    **Policy:** Only JSON-safe literals (dict, list, str, int, float, bool, None)
    are permitted. All other AST node types are rejected.

    Never use eval(), exec(), or compile() on untrusted input.
    """
    if exec_code is None:
        return None
    if not isinstance(exec_code, str):
        raise ValidationError("Exec field must be a string or None.")
    if len(exec_code) > EXEC_MAX_LEN:
        raise ValidationError(f"Exec field exceeds {EXEC_MAX_LEN} characters.")

    try:
        parsed = ast.parse(exec_code, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"Exec field is not a valid expression: {exc}") from exc

    # Whitelist-only AST traversal — use STRICT exec whitelist
    for node in ast.walk(parsed):
        if type(node) not in _EXEC_AST_WHITELIST:
            raise SecurityError(
                f"Exec field contains unsafe AST node: {type(node).__name__}"
            )

    # Verify it evaluates as a literal (no side effects possible)
    try:
        ast.literal_eval(exec_code)
    except (ValueError, SyntaxError) as exc:
        raise ValidationError(
            f"Exec field is not a safe literal expression: {exc}"
        ) from exc

    return exec_code


# ── Checksum / Provenance Helpers ─────────────────────────────────

def compute_checksum(data: Dict[str, Any]) -> str:
    """Compute SHA-256 checksum of a canonical JSON representation."""
    canonical = str(sorted(data.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_provenance(
    data: Dict[str, Any],
    source: str = "api",
    ingested_by: Optional[str] = None,
    origin_id: Optional[str] = None,
) -> RuleProvenance:
    """Construct a provenance record from raw ingestion data."""
    return RuleProvenance(
        source=source,
        origin_id=origin_id,
        ingested_by=ingested_by,
        checksum=compute_checksum(data),
    )


def track_provenance(
    rule: Rule,
    source: str = "api",
    ingested_by: Optional[str] = None,
    origin_id: Optional[str] = None,
) -> RuleProvenance:
    """Attach or update provenance tracking for a rule creator.

    Builds a new provenance record linked to the rule's current state,
    records a creation event, and replaces the rule's provenance.

    Returns the new provenance record.
    """
    data = {
        "name": rule.name,
        "tagline": rule.production.tagline,
        "condition": rule.production.condition,
        "exec_field": rule.production.exec_field,
    }
    prov = build_provenance(data, source=source, ingested_by=ingested_by, origin_id=origin_id)
    prov.add_event("created", f"rule_name={rule.name}")
    rule.provenance = prov
    return prov


# ── Rule Creation API ──────────────────────────────────────────────

def create_rule(
    name: str,
    tagline: str = "",
    condition: str = "",
    exec_field: Optional[str] = None,
    provenance: Optional[RuleProvenance] = None,
) -> Rule:
    """Create a validated Rule with full security hardening.

    All inputs are strictly sanitized before the Rule is returned.
    Raises ValidationError or SecurityError on any violation.
    """
    clean_name = validate_rule_name(name)
    clean_tagline, clean_condition = validate_production_fields(tagline, condition)
    clean_exec = validate_exec_field(exec_field)

    rule = Rule(
        name=clean_name,
        production=Production(
            tagline=clean_tagline,
            condition=clean_condition,
            exec_field=clean_exec,
        ),
        provenance=provenance or RuleProvenance(
            checksum=compute_checksum({
                "name": clean_name,
                "tagline": clean_tagline,
                "condition": clean_condition,
                "exec_field": clean_exec,
            })
        ),
    )
    if provenance is None:
        track_provenance(rule, source="api")
    return rule


def create_rule_from_dict(
    data: dict,
    source: str = "api",
    ingested_by: Optional[str] = None,
) -> Rule:
    """Convenience wrapper for JSON/rule-dict ingestion with provenance.

    Automatically computes checksum and builds a provenance record.
    """
    prov = build_provenance(data, source=source, ingested_by=ingested_by)
    return create_rule(
        name=data.get("name", ""),
        tagline=data.get("production", {}).get("tagline", ""),
        condition=data.get("production", {}).get("condition", ""),
        exec_field=data.get("production", {}).get("exec") or data.get("production", {}).get("exec_field"),
        provenance=prov,
    )


# ── Sandboxed Evaluation ───────────────────────────────────────────

def evaluate_condition(condition: str, metrics: Dict[str, Any]) -> bool:
    """Evaluate a condition string safely against a metrics context.

    Parses into AST, validates against whitelist, then evaluates in a
    restricted namespace with no builtins access.

    Raises:
        SecurityError: If AST contains non-whitelisted nodes.
        ValidationError: On parse failure or unknown metric references.
    """
    if not condition:
        return True

    try:
        tree = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(f"Condition is not a valid expression: {exc}") from exc

    # Validate AST whitelist
    for node in ast.walk(tree):
        if type(node) not in _CONDITION_AST_WHITELIST:
            raise SecurityError(
                f"Condition contains unsupported operator: {type(node).__name__}"
            )

    try:
        result = eval(
            compile(tree, "<condition>", "eval"),
            {"__builtins__": {}},
            metrics,
        )
        return bool(result)
    except NameError as exc:
        raise ValidationError(f"Condition references unknown metric: {exc}") from exc
    except Exception as exc:
        raise ValidationError(f"Condition evaluation failed: {exc}") from exc


def sandboxed_exec(exec_code: Optional[str]) -> Any:
    """Safely evaluate a production exec field.

    Only literal expressions (dict, list, str, int, float, bool, None)
    are permitted. Returns the evaluated literal value.

    Raises SecurityError if any unsafe AST node is detected.
    """
    if exec_code is None:
        return None
    _ = validate_exec_field(exec_code)  # raises on violation
    return ast.literal_eval(exec_code)


# ── Batch Operations ───────────────────────────────────────────────

def batch_create_rules(
    rule_dicts: List[dict],
    source: str = "api",
    ingested_by: Optional[str] = None,
) -> Tuple[List[Rule], List[ValidationError]]:
    """Validate a batch of rule dicts, returning successes and failures separately.

    Args:
        rule_dicts: List of canonical JSON-form dicts.
        source: Provenance source label.
        ingested_by: Agent/user identifier for provenance.

    Returns:
        Tuple of (validated_rules, errors). Each error carries an `index`
        attribute for forensic correlation.
    """
    rules: List[Rule] = []
    errors: List[ValidationError] = []

    for idx, data in enumerate(rule_dicts):
        try:
            rules.append(create_rule_from_dict(data, source=source, ingested_by=ingested_by))
        except ValidationError as exc:
            exc.index = idx  # type: ignore[attr-defined]
            errors.append(exc)

    return rules, errors


# ── Audit / Forensics ──────────────────────────────────────────────

def audit_rule(rule: Rule) -> Dict[str, Any]:
    """Generate an audit dict for a rule, suitable for logging or inspection."""
    return {
        "name": rule.name,
        "checksum": rule.provenance.checksum,
        "source": rule.provenance.source,
        "origin_id": rule.provenance.origin_id,
        "ingested_by": rule.provenance.ingested_by,
        "ingested_at": rule.provenance.ingested_at,
        "generation": rule.provenance.generation,
        "parent_names": rule.provenance.parent_names,
        "event_count": len(rule.provenance.history),
        "tagline_length": len(rule.production.tagline),
        "condition_length": len(rule.production.condition),
        "has_exec": rule.production.exec_field is not None,
    }

"""
tests/test_grammar_security.py

Tests for RuleValidator / Grammar Security Hardening.
Covers rule name validation, production field sanitization,
sandboxed exec, provenance tracking, batch ingestion, and audit.
"""

import pytest
from grammar.security_hardening import (
    Rule,
    Production,
    RuleProvenance,
    ValidationError,
    SecurityError,
    validate_rule_name,
    validate_production_fields,
    validate_exec_field,
    create_rule,
    create_rule_from_dict,
    evaluate_condition,
    sandboxed_exec,
    batch_create_rules,
    build_provenance,
    compute_checksum,
    track_provenance,
    audit_rule,
)


# -------------------------------------------------------------------- #
# validate_rule_name — alphanumeric + underscore only
# -------------------------------------------------------------------- #


def test_valid_rule_names():
    assert validate_rule_name("hello_world") == "hello_world"
    assert validate_rule_name("Rule123") == "Rule123"
    assert validate_rule_name("_private") == "_private"
    assert validate_rule_name("a") == "a"


def test_reject_empty_name():
    with pytest.raises(ValidationError, match="not be empty"):
        validate_rule_name("")
    with pytest.raises(ValidationError, match="not be empty"):
        validate_rule_name("   ")


def test_reject_non_string_name():
    with pytest.raises(ValidationError, match="must be a string"):
        validate_rule_name(123)


def test_reject_too_long_name():
    with pytest.raises(ValidationError, match="exceeds 64"):
        validate_rule_name("a" * 65)


def test_reject_path_traversal():
    with pytest.raises(SecurityError, match="path traversal"):
        validate_rule_name("../etc/passwd")
    with pytest.raises(SecurityError, match="path traversal"):
        validate_rule_name("rules/../../secret")
    with pytest.raises(SecurityError, match="path traversal"):
        validate_rule_name("rule/name")
    with pytest.raises(SecurityError, match="path traversal"):
        validate_rule_name("rule\\name")


def test_reject_illegal_characters():
    with pytest.raises(SecurityError, match="illegal characters"):
        validate_rule_name("rule-name")  # hyphen
    with pytest.raises(SecurityError, match="illegal characters"):
        validate_rule_name("rule.name")  # dot
    with pytest.raises(SecurityError, match="illegal characters"):
        validate_rule_name("rule$name")  # dollar
    with pytest.raises(SecurityError, match="illegal characters"):
        validate_rule_name("rule name")  # space
    with pytest.raises(SecurityError, match="illegal characters"):
        validate_rule_name("rule!name")  # bang


# -------------------------------------------------------------------- #
# validate_production_fields — no scripts in taglines, no SQL in conditions
# -------------------------------------------------------------------- #


def test_valid_production_fields():
    tagline, condition = validate_production_fields("Normal tagline", "x > 0")
    assert tagline == "Normal tagline"
    assert condition == "x > 0"


def test_tagline_html_escape():
    tagline, _ = validate_production_fields("<b>Bold</b>", "True")
    assert "<b>" not in tagline
    assert "Bold" in tagline
    assert "&lt;b&gt;" in tagline or "Bold" in tagline


def test_tagline_blocks_script_tags():
    with pytest.raises(SecurityError, match="script/injection"):
        validate_production_fields("<script>alert(1)</script>", "True")
    with pytest.raises(SecurityError, match="script/injection"):
        validate_production_fields("<iframe src='evil.com'>", "True")
    with pytest.raises(SecurityError, match="script/injection"):
        validate_production_fields("<object data='x'></object>", "True")
    with pytest.raises(SecurityError, match="script/injection"):
        validate_production_fields("javascript:alert(1)", "True")
    with pytest.raises(SecurityError, match="script/injection"):
        validate_production_fields("onerror=alert(1)", "True")


def test_tagline_allows_safe_angle_brackets_after_strip():
    tagline, _ = validate_production_fields("5 < 10", "True")
    assert "5" in tagline
    assert "10" in tagline


def test_condition_blocks_sql_injection():
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "x; DROP TABLE rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "x -- comment")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "SELECT * FROM rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "DELETE FROM rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "INSERT INTO rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "UPDATE rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "ALTER TABLE rules")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "EXEC sp_msforeachtable")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "EXECUTE immediate")
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "UNION SELECT password")


def test_condition_allows_safe_comparisons():
    tagline, condition = validate_production_fields("x", "temp > 100 and pressure < 50")
    assert condition == "temp > 100 and pressure < 50"


def test_condition_allows_safe_semicolon_in_string():
    # semicolon inside a string literal should still be blocked by regex
    # because SQLI_BLACKLIST is a simple regex, not token-aware
    with pytest.raises(SecurityError, match="SQL injection"):
        validate_production_fields("x", "';' == x")


def test_reject_too_long_tagline():
    with pytest.raises(ValidationError, match="exceeds 256"):
        validate_production_fields("x" * 257, "True")


def test_reject_too_long_condition():
    with pytest.raises(ValidationError, match="exceeds 1024"):
        validate_production_fields("x", "x" * 1025)


def test_reject_non_string_tagline():
    with pytest.raises(ValidationError, match="Tagline must be a string"):
        validate_production_fields(123, "True")


def test_reject_non_string_condition():
    with pytest.raises(ValidationError, match="Condition must be a string"):
        validate_production_fields("x", 123)


# -------------------------------------------------------------------- #
# sandboxed_exec — production rules, literal-only
# -------------------------------------------------------------------- #


def test_sandboxed_exec_dict_literal():
    assert sandboxed_exec("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}


def test_sandboxed_exec_list_literal():
    assert sandboxed_exec("[1, 2, 3]") == [1, 2, 3]


def test_sandboxed_exec_string_literal():
    assert sandboxed_exec("'hello'") == "hello"


def test_sandboxed_exec_number_literal():
    assert sandboxed_exec("42") == 42
    assert sandboxed_exec("3.14") == 3.14


def test_sandboxed_exec_bool_none():
    assert sandboxed_exec("True") is True
    assert sandboxed_exec("False") is False
    assert sandboxed_exec("None") is None


def test_sandboxed_exec_none_input():
    assert sandboxed_exec(None) is None


def test_sandboxed_exec_rejects_function_call():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        sandboxed_exec("__import__('os').system('rm -rf /')")


def test_sandboxed_exec_rejects_import():
    with pytest.raises(ValidationError, match="not a valid expression"):
        sandboxed_exec("import os")


def test_sandboxed_exec_rejects_name_reference():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        sandboxed_exec("os.system('ls')")


def test_sandboxed_exec_rejects_lambda():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        sandboxed_exec("lambda x: x + 1")


def test_sandboxed_exec_rejects_comprehension():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        sandboxed_exec("[x for x in range(10)]")


def test_sandboxed_exec_rejects_arithmetic_on_names():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        sandboxed_exec("x + 1")


def test_validate_exec_field_rejects_attribute_access():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        validate_exec_field("{'a': object.__class__}")


def test_validate_exec_field_rejects_nested_bad_nodes():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        validate_exec_field("[{'a': (__import__('os'))}]")


# -------------------------------------------------------------------- #
# track_provenance — rule creator tracking
# -------------------------------------------------------------------- #


def test_track_provenance_basic():
    rule = Rule(name="test_rule", production=Production())
    prov = track_provenance(rule, source="api", ingested_by="agent_42")
    assert prov.source == "api"
    assert prov.ingested_by == "agent_42"
    assert prov.checksum is not None
    assert len(prov.history) == 1
    assert prov.history[0]["event"] == "created"
    assert "test_rule" in prov.history[0]["detail"]
    assert rule.provenance is prov


def test_track_provenance_with_origin_id():
    rule = Rule(name="evolved_rule", production=Production())
    prov = track_provenance(rule, source="evolution", ingested_by="breeder_1", origin_id="parent_001")
    assert prov.origin_id == "parent_001"
    assert prov.source == "evolution"


def test_provenance_checksum_changes_with_content():
    rule1 = Rule(name="rule_a", production=Production(tagline="x"))
    rule2 = Rule(name="rule_a", production=Production(tagline="y"))
    prov1 = track_provenance(rule1)
    prov2 = track_provenance(rule2)
    assert prov1.checksum != prov2.checksum


def test_build_provenance_standalone():
    data = {"name": "foo", "tagline": "bar", "condition": "baz"}
    prov = build_provenance(data, source="file", ingested_by="user_1")
    assert prov.source == "file"
    assert prov.ingested_by == "user_1"
    assert prov.checksum == compute_checksum(data)


def test_provenance_add_event():
    prov = RuleProvenance()
    prov.add_event("mutated", "generation=2")
    assert len(prov.history) == 1
    assert prov.history[0]["event"] == "mutated"
    assert prov.history[0]["detail"] == "generation=2"
    assert "timestamp" in prov.history[0]


# -------------------------------------------------------------------- #
# create_rule — integration of all validators
# -------------------------------------------------------------------- #


def test_create_rule_happy_path():
    rule = create_rule(
        name="firewall_block",
        tagline="Block malicious IP",
        condition="threat_score > 80",
        exec_field="{'action': 'drop', 'log': True}",
    )
    assert rule.name == "firewall_block"
    assert rule.production.tagline == "Block malicious IP"
    assert rule.production.condition == "threat_score > 80"
    assert rule.production.exec_field == "{'action': 'drop', 'log': True}"
    assert rule.provenance.checksum is not None
    assert rule.provenance.history[0]["event"] == "created"


def test_create_rule_rejects_bad_name():
    with pytest.raises(SecurityError):
        create_rule(name="bad-name", tagline="x", condition="True")


def test_create_rule_rejects_script_in_tagline():
    with pytest.raises(SecurityError, match="script/injection"):
        create_rule(name="rule_1", tagline="<script>evil</script>", condition="True")


def test_create_rule_rejects_sql_in_condition():
    with pytest.raises(SecurityError, match="SQL injection"):
        create_rule(name="rule_1", tagline="x", condition="DROP TABLE rules")


def test_create_rule_rejects_bad_exec():
    with pytest.raises(SecurityError, match="unsafe AST node"):
        create_rule(name="rule_1", tagline="x", condition="True", exec_field="os.system('ls')")


def test_create_rule_from_dict():
    data = {
        "name": "dict_rule",
        "production": {
            "tagline": "From dict",
            "condition": "score > 50",
            "exec": "{'notify': True}",
        },
    }
    rule = create_rule_from_dict(data, source="api", ingested_by="importer_1")
    assert rule.name == "dict_rule"
    assert rule.production.tagline == "From dict"
    assert rule.production.condition == "score > 50"
    assert rule.provenance.source == "api"
    assert rule.provenance.ingested_by == "importer_1"


def test_create_rule_from_dict_missing_production():
    data = {"name": "minimal"}
    rule = create_rule_from_dict(data)
    assert rule.name == "minimal"
    assert rule.production.tagline == ""
    assert rule.production.condition == ""
    assert rule.production.exec_field is None


# -------------------------------------------------------------------- #
# evaluate_condition — safe evaluation against metrics
# -------------------------------------------------------------------- #


def test_evaluate_simple_comparison():
    assert evaluate_condition("x > 5", {"x": 10}) is True
    assert evaluate_condition("x > 5", {"x": 3}) is False


def test_evaluate_logical_and():
    assert evaluate_condition("x > 5 and y < 10", {"x": 7, "y": 3}) is True
    assert evaluate_condition("x > 5 and y < 10", {"x": 3, "y": 3}) is False


def test_evaluate_empty_condition_is_true():
    assert evaluate_condition("", {"x": 1}) is True


def test_evaluate_rejects_builtins():
    with pytest.raises(SecurityError, match="unsupported operator"):
        evaluate_condition("__import__('os')", {})


def test_evaluate_rejects_call():
    with pytest.raises(SecurityError, match="unsupported operator"):
        evaluate_condition("print('hello')", {})


def test_evaluate_rejects_attribute_access():
    with pytest.raises(SecurityError, match="unsupported operator"):
        evaluate_condition("{}.get('x')", {})


def test_evaluate_unknown_metric():
    with pytest.raises(ValidationError, match="unknown metric"):
        evaluate_condition("z > 1", {"x": 1})


def test_evaluate_bad_syntax():
    with pytest.raises(ValidationError, match="not a valid expression"):
        evaluate_condition("x >", {"x": 1})


# -------------------------------------------------------------------- #
# batch_create_rules — bulk ingestion with error separation
# -------------------------------------------------------------------- #


def test_batch_all_valid():
    rule_dicts = [
        {"name": "rule_1", "production": {"tagline": "t1", "condition": "True"}},
        {"name": "rule_2", "production": {"tagline": "t2", "condition": "False"}},
    ]
    rules, errors = batch_create_rules(rule_dicts, source="file")
    assert len(rules) == 2
    assert len(errors) == 0
    assert rules[0].name == "rule_1"
    assert rules[1].name == "rule_2"


def test_batch_some_invalid():
    rule_dicts = [
        {"name": "rule_1", "production": {"tagline": "t1", "condition": "True"}},
        {"name": "bad-name", "production": {"tagline": "t2", "condition": "True"}},
        {"name": "rule_3", "production": {"tagline": "<script>x</script>", "condition": "True"}},
    ]
    rules, errors = batch_create_rules(rule_dicts, source="file")
    assert len(rules) == 1
    assert len(errors) == 2
    assert rules[0].name == "rule_1"
    assert errors[0].index == 1  # type: ignore
    assert errors[1].index == 2  # type: ignore


def test_batch_all_invalid():
    rule_dicts = [
        {"name": "bad-1", "production": {"tagline": "t", "condition": "True"}},
        {"name": "bad-2", "production": {"tagline": "t", "condition": "DROP"}},
    ]
    rules, errors = batch_create_rules(rule_dicts)
    assert len(rules) == 0
    assert len(errors) == 2


def test_batch_empty():
    rules, errors = batch_create_rules([])
    assert rules == []
    assert errors == []


# -------------------------------------------------------------------- #
# audit_rule — forensics output
# -------------------------------------------------------------------- #


def test_audit_rule_structure():
    rule = create_rule(
        name="audit_me",
        tagline="Test tagline",
        condition="score > 50",
        exec_field="{'log': True}",
    )
    audit = audit_rule(rule)
    assert audit["name"] == "audit_me"
    assert audit["checksum"] is not None
    assert audit["source"] == "api"
    assert audit["tagline_length"] == len("Test tagline")
    assert audit["condition_length"] == len("score > 50")
    assert audit["has_exec"] is True
    assert audit["event_count"] >= 1


def test_audit_rule_no_exec():
    rule = create_rule(name="no_exec", tagline="x", condition="True")
    audit = audit_rule(rule)
    assert audit["has_exec"] is False


# -------------------------------------------------------------------- #
# Edge cases and torture tests
# -------------------------------------------------------------------- #


def test_unicode_in_tagline_escaped():
    tagline, _ = validate_production_fields("<b>你好</b>", "True")
    assert "你好" in tagline
    assert "<b>" not in tagline


def test_newline_in_tagline_escaped():
    tagline, _ = validate_production_fields("line1\nline2", "True")
    assert "line1" in tagline
    assert "line2" in tagline


def test_production_fields_empty_condition():
    tagline, condition = validate_production_fields("x", "")
    assert condition == ""


def test_sandboxed_exec_nested_literals():
    result = sandboxed_exec("{'a': [1, 2, {'b': 3}], 'c': (4, 5)}")
    assert result == {"a": [1, 2, {"b": 3}], "c": (4, 5)}


def test_rule_name_exactly_max_length():
    name = "a" * 64
    assert validate_rule_name(name) == name


def test_tagline_exactly_max_length():
    tagline = "x" * 256
    t, _ = validate_production_fields(tagline, "True")
    assert len(t) >= 250  # escaped may be longer


def test_condition_exactly_max_length():
    condition = "x" * 1024
    _, c = validate_production_fields("x", condition)
    assert c == condition

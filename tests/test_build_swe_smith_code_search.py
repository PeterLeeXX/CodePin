from unidiff import PatchSet

from src.build_swe_smith_code_search import (
    apply_file_patch,
    classify_ineligible,
    file_change_from_patch,
)


def _single_file_patch(patch: str):
    patch_set = PatchSet(patch)
    assert len(patch_set) == 1
    return patch_set[0]


def test_applies_patch_with_no_newline_marker():
    source = "def answer():\n    return 41"
    patch = """diff --git a/example.py b/example.py
--- a/example.py
+++ b/example.py
@@ -1,2 +1,2 @@
 def answer():
-    return 41
+    return 42
\\ No newline at end of file
"""
    patched_file = _single_file_patch(patch)

    assert apply_file_patch(source, patched_file) == "def answer():\n    return 42\n"


def test_deleted_buggy_patch_method_is_added_by_inverse_fix():
    source = """class Service:
    def keep(self):
        return True

    def restore_me(self):
        return 1
"""
    patch = """diff --git a/service.py b/service.py
--- a/service.py
+++ b/service.py
@@ -2,5 +2,2 @@ class Service:
     def keep(self):
         return True
-
-    def restore_me(self):
-        return 1
"""
    patched_file = _single_file_patch(patch)
    patched = apply_file_patch(source, patched_file)

    result = file_change_from_patch("service.py", source, patched, patched_file)

    assert result["changes"]["added_entities"] == [
        "service.py:Service.restore_me"
    ]
    assert result["changes"]["edited_modules"] == ["service.py:Service"]


def test_inline_comment_does_not_hide_function_edit():
    source = """def values(items):
    return sorted(items, key=len)  # preserve stable order
"""
    patch = """diff --git a/example.py b/example.py
--- a/example.py
+++ b/example.py
@@ -1,2 +1,2 @@
 def values(items):
-    return sorted(items, key=len)  # preserve stable order
+    return sorted(items, reverse=True)  # preserve stable order
"""
    patched_file = _single_file_patch(patch)
    patched = apply_file_patch(source, patched_file)

    result = file_change_from_patch("example.py", source, patched, patched_file)

    assert result["changes"]["edited_entities"] == ["example.py:values"]
    assert result["changes"]["edited_modules"] == ["example.py:values"]


def test_cleaning_filters_empty_non_python_and_created_files():
    base = {"repo": "swesmith/example__example.123", "problem_statement": "bug"}
    python_edit = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
    created_file = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+x = 1
"""
    text_edit = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""

    reason, keys = classify_ineligible({**base, "patch": python_edit})
    assert reason is None
    assert [(key.repo, key.path) for key in keys] == [
        ("swesmith/example__example.123", "a.py")
    ]

    reason, _ = classify_ineligible(
        {**base, "problem_statement": "  ", "patch": python_edit}
    )
    assert reason == "empty_problem_statement"
    assert classify_ineligible({**base, "patch": created_file})[0] == (
        "creates_or_deletes_file"
    )
    assert classify_ineligible({**base, "patch": text_edit})[0] == (
        "no_python_changes"
    )

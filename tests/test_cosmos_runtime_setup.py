"""Repeated preparation must preserve the installed Torch compatibility fixes."""
import pytest
from tools.setup_cosmos_pointfk_runtime import apply_compatibility_patch


PATCH = '--- a/source/example.py\n+++ b/source/example.py\n@@ -1 +1 @@\n-old\n+compatible\n'


def test_compatibility_patch_is_idempotent(tmp_path):
    target = tmp_path / 'source/example.py'
    target.parent.mkdir()
    target.write_text('old\n')
    for _ in range(3):
        apply_compatibility_patch(tmp_path, PATCH)
        assert target.read_text() == 'compatible\n'


def test_incompatible_source_is_rejected_without_mutation(tmp_path):
    target = tmp_path / 'source/example.py'
    target.parent.mkdir()
    target.write_text('unexpected version\n')
    with pytest.raises(RuntimeError, match='differs'):
        apply_compatibility_patch(tmp_path, PATCH)
    assert target.read_text() == 'unexpected version\n'

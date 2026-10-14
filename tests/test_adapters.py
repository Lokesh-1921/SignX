import pytest

import datasets.adapters  # noqa: F401 - imports adapter registrations
from datasets.base import adapter_registry


def test_standard_adapters_are_registered():
    """All expected dataset adapters should be in the registry."""
    assert {"wlasl", "msasl", "how2sign", "autsl", "phoenix2014t", "csl"}.issubset(adapter_registry)


def test_stub_adapters_raise_not_implemented(tmp_path):
    """Stub adapters (not yet implemented) should raise NotImplementedError."""
    stubs = {"msasl", "how2sign", "autsl", "phoenix2014t", "csl"}
    for name in stubs:
        with pytest.raises(NotImplementedError):
            list(adapter_registry[name]().build_records(tmp_path))


def test_wlasl_adapter_is_implemented(tmp_path):
    """WLASL adapter should NOT raise NotImplementedError (it's fully implemented)."""
    with pytest.raises(FileNotFoundError, match="WLASL"):
        # Without annotation files it should raise FileNotFoundError, not NotImplementedError
        list(adapter_registry["wlasl"]().build_records(tmp_path))

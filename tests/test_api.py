from app.backend.main import health, models


def test_api_foundation_endpoints():
    assert health() == {"status": "ok", "service": "signbridge"}
    assert models() == []

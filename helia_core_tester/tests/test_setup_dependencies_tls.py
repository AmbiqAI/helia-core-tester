"""TLS context used for dependency downloads."""

import hashlib
import ssl
from unittest.mock import patch

import certifi
import pytest

from helia_core_tester.scripts import setup_dependencies as sd


@pytest.fixture
def no_cert_env(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def _loaded_cafiles(monkeypatch) -> list:
    calls = []
    real = ssl.SSLContext.load_verify_locations

    def spy(self, cafile=None, *args, **kwargs):
        calls.append(cafile)
        return real(self, cafile, *args, **kwargs)

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", spy)
    return calls


def test_tls_context_verifies(no_cert_env):
    ctx = sd.tls_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.cert_store_stats()["x509_ca"] > 0


def test_tls_context_adds_certifi(no_cert_env, monkeypatch):
    calls = _loaded_cafiles(monkeypatch)
    sd.tls_context()
    assert certifi.where() in calls


@pytest.mark.parametrize("var", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_tls_context_honours_env(var, monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv(var, str(tmp_path / "unused"))
    calls = _loaded_cafiles(monkeypatch)
    ctx = sd.tls_context()
    assert certifi.where() not in calls
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_download_passes_context(tmp_path, no_cert_env):
    payload = b"payload"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            nonlocal payload
            chunk, payload = payload, b""
            return chunk

    with patch("urllib.request.urlopen", return_value=_Resp()) as urlopen:
        sd.download_file(
            "https://example.invalid/f",
            tmp_path / "f.bin",
            "test file",
            hashlib.sha256(b"payload").hexdigest(),
        )
    ctx = urlopen.call_args.kwargs["context"]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED

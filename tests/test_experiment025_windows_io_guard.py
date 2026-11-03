"""Mock-only regression tests for frozen-source Experiment025 VM I/O recovery."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(script: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    spec = importlib.util.spec_from_file_location(script.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def access_denied(winerror: int = 5) -> PermissionError:
    error = PermissionError(13, "Access is denied")
    error.winerror = winerror
    return error


def test_transient_winerror5_retries_same_atomic_replace_without_data_loss():
    guard = load_script("experiment025-windows-io-guard.py")
    src = (r"results\gset\experiment025-gw16-bestcut-quality\full"
           r"\G21\control-paired\progress\0002\best.json.tmp")
    dst = src.removesuffix(".tmp")
    attempts, delays = [], []

    def transient(s, t):
        attempts.append((s, t))
        if len(attempts) < 3:
            raise access_denied(5)

    guard.retry_checkpoint_replace(
        src, dst, replace_func=transient, sleep_func=delays.append,
        attempts=4)
    assert attempts == [(src, dst)] * 3
    assert delays == [0.05, 0.1]


def test_winerror32_sharing_violation_is_retryable():
    guard = load_script("experiment025-windows-io-guard.py")
    src = "results/gset/experiment025-gw16-bestcut-quality/full/G10/progress/best.json.tmp"
    dst = src.removesuffix(".tmp")
    calls = []

    def transient(*args):
        calls.append(args)
        if len(calls) == 1:
            raise access_denied(32)

    guard.retry_checkpoint_replace(
        src, dst, replace_func=transient, sleep_func=lambda _: None)
    assert len(calls) == 2


def test_other_io_errors_propagate_without_retry():
    guard = load_script("experiment025-windows-io-guard.py")
    src = "results/gset/experiment025-gw16-bestcut-quality/full/G20/best.json.tmp"
    dst = src.removesuffix(".tmp")
    calls = []

    def wrong(*args):
        calls.append(args)
        raise access_denied(123)

    with pytest.raises(PermissionError):
        guard.retry_checkpoint_replace(
            src, dst, replace_func=wrong, sleep_func=lambda _: None)
    assert len(calls) == 1


def test_unrelated_path_does_not_retry_even_on_winerror5():
    guard = load_script("experiment025-windows-io-guard.py")
    calls = []

    def denied(*args):
        calls.append(args)
        raise access_denied()

    with pytest.raises(PermissionError):
        guard.retry_checkpoint_replace(
            "data/unrelated.tmp", "data/unrelated.json",
            replace_func=denied, sleep_func=lambda _: None)
    assert len(calls) == 1


def test_bounded_retry_exhaustion_raises_and_preserves_temp_file():
    guard = load_script("experiment025-windows-io-guard.py")
    calls, delays = [], []
    src = "results/gset/experiment025-gw16-bestcut-quality/full/G21/best.json.tmp"
    dst = src.removesuffix(".tmp")

    def always_locked(*args):
        calls.append(args)
        raise access_denied()

    with pytest.raises(PermissionError):
        guard.retry_checkpoint_replace(
            src, dst, replace_func=always_locked,
            sleep_func=delays.append, attempts=3)
    assert len(calls) == 3
    assert delays == [0.05, 0.1]


def test_ambiguous_git_push_considers_exact_remote_sha_a_receipt(monkeypatch):
    driver = load_script("experiment025-parallel.py")
    calls = []

    def git(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("rev-parse", "HEAD"):
            return SimpleNamespace(stdout="abcdef\n")
        if args[0] == "push":
            raise RuntimeError("RPC failed; curl 55")
        if args[0] == "ls-remote":
            return SimpleNamespace(stdout="abcdef\trefs/heads/experiment025-results\n")
        raise AssertionError(args)

    monkeypatch.setattr(driver, "git", git)
    driver.push_with_confirmation()
    assert sum(args[0] == "push" for args in calls) == 1


def test_git_nonfastforward_rejects_without_retry(monkeypatch):
    driver = load_script("experiment025-parallel.py")
    attempts = []

    def git(*args, **kwargs):
        if args[:2] == ("rev-parse", "HEAD"):
            return SimpleNamespace(stdout="abcdef\n")
        if args[0] == "push":
            attempts.append(args)
            raise RuntimeError("non-fast-forward")
        if args[0] == "ls-remote":
            return SimpleNamespace(stdout="def123\trefs/heads/experiment025-results\n")
        raise AssertionError(args)

    monkeypatch.setattr(driver, "git", git)
    with pytest.raises(RuntimeError, match="non-fast-forward"):
        driver.push_with_confirmation()
    assert len(attempts) == 1

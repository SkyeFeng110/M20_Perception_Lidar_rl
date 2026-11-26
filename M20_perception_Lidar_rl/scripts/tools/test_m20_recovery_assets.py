"""Recovery asset overrides must work without historical replay downloads."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def assets(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2]/'source/rl_training/rl_training/recovery_assets.py'
    spec = importlib.util.spec_from_file_location('m20_recovery_assets_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'DATA', tmp_path/'defaults')
    for name in ('M20_RECOVERY_TEACHER', 'M20_RECOVERY_REPLAY', 'M20_BASE_CHECKPOINT'):
        monkeypatch.delenv(name, raising=False)
    return module


def test_complete_overrides_do_not_require_default_metadata(assets, monkeypatch):
    monkeypatch.setenv('M20_RECOVERY_TEACHER', 'teacher/model_58098.pt')
    monkeypatch.setenv('M20_RECOVERY_REPLAY', 'stage/current_gait.npz')
    assert assets.resolve_assets() == (assets.ROOT/'teacher/model_58098.pt', assets.ROOT/'stage/current_gait.npz')


def test_complete_overrides_ignore_unrelated_invalid_default(assets, monkeypatch):
    assets.DATA.mkdir()
    (assets.DATA/'reference_observations_v2.json').write_text('invalid old metadata')
    monkeypatch.setenv('M20_BASE_CHECKPOINT', 'base.pt')
    monkeypatch.setenv('M20_RECOVERY_TEACHER', 'preferred.pt')
    monkeypatch.setenv('M20_RECOVERY_REPLAY', 'current.npz')
    assert assets.resolve_assets() == (assets.ROOT/'preferred.pt', assets.ROOT/'current.npz')


def test_base_checkpoint_alias_is_self_contained(assets, monkeypatch):
    monkeypatch.setenv('M20_BASE_CHECKPOINT', 'base.pt')
    monkeypatch.setenv('M20_RECOVERY_REPLAY', 'current.npz')
    assert assets.resolve_assets() == (assets.ROOT/'base.pt', assets.ROOT/'current.npz')


@pytest.mark.parametrize('override', [None, 'teacher', 'replay'])
def test_historical_defaults_and_partial_overrides(assets, monkeypatch, override):
    assets.DATA.mkdir()
    (assets.DATA/'reference_observations_v2.json').write_text(json.dumps(
        dict(source_checkpoint='old.pt', replay_path='old.npz')))
    teacher, replay = assets.ROOT/'old.pt', assets.ROOT/'old.npz'
    if override == 'teacher':
        monkeypatch.setenv('M20_RECOVERY_TEACHER', 'new.pt')
        teacher = assets.ROOT/'new.pt'
    if override == 'replay':
        monkeypatch.setenv('M20_RECOVERY_REPLAY', 'new.npz')
        replay = assets.ROOT/'new.npz'
    assert assets.resolve_assets() == (teacher, replay)


def test_missing_partial_default_reports_required_overrides(assets, monkeypatch):
    monkeypatch.setenv('M20_RECOVERY_TEACHER', 'teacher.pt')
    with pytest.raises(FileNotFoundError, match='Set both M20_RECOVERY_TEACHER and M20_RECOVERY_REPLAY'):
        assets.resolve_assets()

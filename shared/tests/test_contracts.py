"""Step 1 gate: every contract round-trips Python <-> JSON, fixtures validate against the exported JSON Schema."""

import json
from typing import Any

import pytest
from jsonschema import Draft7Validator
from pydantic import BaseModel, ValidationError

from shared.schemas import ROOT_MODELS, MatchEvent, OverlayCue
from shared.schemas.events import QUALIFIER_KIND
from shared.schemas.export import FIXTURE_DIR, JSON_DIR, outputs
from shared.schemas.fixtures import FIXTURES, extra_events, match_event, overlay_cue
from shared.schemas.models import PitchControlGrid

MODELS = {m.__name__: m for m in ROOT_MODELS}


def test_every_root_model_has_a_fixture() -> None:
    assert set(FIXTURES) == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_python_json_round_trip(name: str) -> None:
    model = MODELS[name]
    original = FIXTURES[name]
    as_json = original.model_dump_json()
    restored = model.model_validate_json(as_json)
    assert restored == original
    assert restored.model_dump_json() == as_json


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fixture_file_matches_and_validates(name: str) -> None:
    data = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    assert MODELS[name].model_validate(data) == FIXTURES[name]
    schema = json.loads((JSON_DIR / f"{name}.schema.json").read_text())
    Draft7Validator.check_schema(schema)
    errors = sorted(Draft7Validator(schema).iter_errors(data), key=str)
    assert not errors, errors[:3]


def test_bundle_schema_validates_all_fixtures() -> None:
    bundle = json.loads((JSON_DIR / "contracts.schema.json").read_text())
    Draft7Validator.check_schema(bundle)
    payload = {name: model.model_dump(mode="json") for name, model in FIXTURES.items()}
    errors = list(Draft7Validator(bundle).iter_errors(payload))
    assert not errors, errors[:3]


def test_json_schema_rejects_what_pydantic_rejects() -> None:
    schema = json.loads((JSON_DIR / "MatchEvent.schema.json").read_text())
    bad: dict[str, Any] = match_event().model_dump(mode="json")
    bad["player"] = "player-nine"  # violates the PlayerId pattern
    assert list(Draft7Validator(schema).iter_errors(bad))
    with pytest.raises(ValidationError):
        MatchEvent.model_validate(bad)
    bad = match_event().model_dump(mode="json")
    bad["location"] = [1.0, 2.0]  # world locations are 3-tuples
    assert list(Draft7Validator(schema).iter_errors(bad))
    with pytest.raises(ValidationError):
        MatchEvent.model_validate(bad)


def test_generated_files_are_current() -> None:
    stale = [str(p) for p, text in outputs().items() if not p.exists() or p.read_text() != text]
    assert not stale, f"run `uv run python -m shared.schemas.export`: {stale}"


@pytest.mark.parametrize("event", extra_events(), ids=lambda e: e.type)
def test_other_event_shapes_round_trip(event: MatchEvent) -> None:
    assert MatchEvent.model_validate_json(event.model_dump_json()) == event


def test_qualifier_table_covers_every_event_type() -> None:
    from typing import get_args

    from shared.schemas.enums import EventType

    assert set(QUALIFIER_KIND) == set(get_args(EventType))


def _with(model: BaseModel, **changes: Any) -> dict[str, Any]:
    data = model.model_dump(mode="json")
    data.update(changes)
    return data


def test_event_qualifiers_must_match_type() -> None:
    shot = match_event()
    with pytest.raises(ValidationError, match="needs qualifiers of kind 'pass'"):
        MatchEvent.model_validate(_with(shot, type="pass"))
    with pytest.raises(ValidationError, match="takes no qualifiers"):
        MatchEvent.model_validate(_with(shot, type="period_start"))
    with pytest.raises(ValidationError, match="needs qualifiers"):
        MatchEvent.model_validate(_with(shot, qualifiers=None))
    # Detected events may omit qualifiers the Spotter could not infer.
    detected = MatchEvent.model_validate(_with(shot, id="ev_d_1", source="detected", confidence=0.6, qualifiers=None))
    assert detected.qualifiers is None
    with pytest.raises(ValidationError, match="no confidence"):
        MatchEvent.model_validate(_with(shot, confidence=0.5))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        MatchEvent.model_validate(_with(match_event(), surprise=1))


def test_pitch_control_size_is_checked() -> None:
    with pytest.raises(ValidationError, match="expected nx\\*ny"):
        PitchControlGrid(team="home", nx=2, ny=2, x0=0, y0=0, cell_m=1, values=[0.5, 0.5, 0.5])


def test_overlay_cue_rules() -> None:
    cue = overlay_cue()
    with pytest.raises(ValidationError, match="t_end"):
        OverlayCue.model_validate(_with(cue, t_end=cue.t_start - 1))
    with pytest.raises(ValidationError, match="anchor kind 'player'"):
        OverlayCue.model_validate(_with(cue, anchor={"kind": "player"}))

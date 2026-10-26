"""Step 1 gate: every contract round-trips Python <-> JSON, fixtures validate against the exported JSON Schema."""

import json
from typing import Any

import pytest
from jsonschema import Draft7Validator
from pydantic import BaseModel, ValidationError

from shared.schemas import ROOT_MODELS, MatchEvent, OverlayCue
from shared.schemas.events import QUALIFIER_KIND
from shared.schemas.export import FIXTURE_DIR, JSON_DIR, outputs
from shared.schemas.fixtures import EXAMPLES, FIXTURES, cue_examples, event_examples, match_event, overlay_cue
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


@pytest.mark.parametrize("event", event_examples(), ids=lambda e: f"{e.type}-{e.source}")
def test_event_examples_round_trip(event: MatchEvent) -> None:
    assert MatchEvent.model_validate_json(event.model_dump_json()) == event


def test_examples_cover_every_variant() -> None:
    from typing import get_args

    from shared.schemas.events import Qualifiers
    from shared.schemas.overlay import CuePayload

    qualifier_kinds = {get_args(m.model_fields["kind"].annotation)[0] for m in get_args(get_args(Qualifiers)[0])}
    assert {e.qualifiers.kind for e in event_examples() if e.qualifiers} == qualifier_kinds
    payload_types = {get_args(m.model_fields["type"].annotation)[0] for m in get_args(get_args(CuePayload)[0])}
    assert {c.payload.type for c in cue_examples()} == payload_types


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_files_validate(name: str) -> None:
    items = json.loads((FIXTURE_DIR / "examples" / f"{name}.json").read_text())
    schema = json.loads((JSON_DIR / f"{name}.schema.json").read_text())
    validator = Draft7Validator(schema)
    for item in items:
        assert MODELS[name].model_validate(item).model_dump(mode="json") == item
        errors = list(validator.iter_errors(item))
        assert not errors, (item["id"], errors[:2])


def test_ts_built_messages_parse_in_pydantic() -> None:
    """TS -> JSON -> Python: messages built with the generated TS types (web/tests/contracts.test.ts)."""
    from pydantic import TypeAdapter

    from shared.schemas import ActionIntent, ViewerProfile
    from shared.schemas.stream import ClientMessage

    data = json.loads((FIXTURE_DIR / "from_ts" / "client_messages.json").read_text())
    assert data["ActionIntent"] and data["ViewerProfile"] and data["ClientMessage"]
    for item in data["ActionIntent"]:
        ActionIntent.model_validate(item)
    for item in data["ViewerProfile"]:
        ViewerProfile.model_validate(item)
    adapter = TypeAdapter(ClientMessage)
    kinds = {adapter.validate_python(item).type for item in data["ClientMessage"]}
    assert kinds == {"play", "pause", "seek", "speed", "profile"}


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
        PitchControlGrid(team="home", nx=2, ny=2, x0=0, y0=0, dx=1, dy=1, values=[0.5, 0.5, 0.5])


def test_overlay_cue_rules() -> None:
    cue = overlay_cue()
    with pytest.raises(ValidationError, match="t_end"):
        OverlayCue.model_validate(_with(cue, t_end=cue.t_start - 1))
    with pytest.raises(ValidationError, match="anchor kind 'player'"):
        OverlayCue.model_validate(_with(cue, anchor={"kind": "player"}))


def test_ground_truth_needs_visible_qualifiers_detected_does_not() -> None:
    shot = match_event()
    data = shot.model_dump(mode="json")
    data["qualifiers"]["technique"] = None
    with pytest.raises(ValidationError, match="needs qualifiers"):
        MatchEvent.model_validate(data)
    data.update(id="ev_d_9", source="detected", confidence=0.7, track_id=4)
    assert MatchEvent.model_validate(data).qualifiers is not None
    with pytest.raises(ValidationError, match="track ids"):
        MatchEvent.model_validate(_with(shot, track_id=3))


def test_shot_outcome_and_on_target_agree() -> None:
    data = match_event().model_dump(mode="json")
    data["qualifiers"]["outcome"] = "off_target"
    with pytest.raises(ValidationError, match="contradicts"):
        MatchEvent.model_validate(data)


def test_rollout_events_are_namespaced() -> None:
    with pytest.raises(ValidationError, match="ev_r_"):
        MatchEvent.model_validate(_with(match_event(), rollout_id="roll_1"))


def test_intent_requires_targets() -> None:
    from shared.schemas import ActionIntent

    with pytest.raises(ValidationError, match="target_player"):
        ActionIntent(kind="pass_player", player="H10")
    with pytest.raises(ValidationError, match="needs target"):
        ActionIntent(kind="shot", player="H10")


def test_stat_keys_resolve() -> None:
    from shared.schemas.fixtures import stats_snapshot
    from shared.schemas.stats import stat_value

    snap = stats_snapshot()
    assert stat_value(snap, "players.H09.top_speed_kmh") == 32.8
    assert stat_value(snap, "teams.home.xg") == 1.12
    with pytest.raises(KeyError):
        stat_value(snap, "players.H99.goals")

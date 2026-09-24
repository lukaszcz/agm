"""TypeSafe System One requests for ``sysone/jev``: AgL values to SDK questions, one pooled
SDK client per settings, SDK answers and failures back to AgL values and ``JevError``s."""

from __future__ import annotations

import dataclasses
import hashlib
import json as pyjson
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from agl import AglException, array, nominals, option_none, option_some, runtime
from agl import dict as agl_dict
from agl import json as agl_json
from typesafe_sdk import (
    ChoiceAnswer,
    ChoiceModel,
    JSONContent,
    NoulAnswer,
    NoulModel,
    QuestionModel,
    RetryPolicy,
    ScoreAnswer,
    ScoreModel,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict
from agm.agl.runtime.serialize import dumps_exact
from agm.core.parse import format_timeout, parse_positive_timeout

if TYPE_CHECKING:
    import httpx2
    from agl import TypeContract

_jev = nominals.sysone.jev
Option = nominals.std.option.Option
TypeError = nominals.std.errors.TypeError

_POOL_KEY = "sysone/jev/client"
# The one question a single-question call sends.
_QUESTION = "question"
_REQUEST_ID_HEADER = "x-typesafe-request-id"
# The most levels a score rubric has.
_MAX_LEVELS = 10
# A noul question's absent criteria.
_NO_CRITERIA = option_none()


@dataclass(frozen=True)
class Settings:
    """The pool key of one client; a ``None`` field leaves the SDK default."""

    api_key: str | None = field(repr=False)
    base_url: str | None
    max_retries: int


def open_client(
    settings: Settings, *, transport: httpx2.BaseTransport | None = None
) -> TypeSafeClient:
    """Build the SDK client for *settings*; tests replace this seam to inject *transport*."""
    configured: dict[str, Any] = {"api_key": settings.api_key, "base_url": settings.base_url}
    return TypeSafeClient(
        **{name: value for name, value in configured.items() if value is not None},
        retry=RetryPolicy(max_retries=settings.max_retries),
        transport=transport,
    )


def _client(settings: Settings) -> TypeSafeClient:
    """The active interpreter's pooled client for *settings*, closed when its run ends."""
    # Hashed so the key never carries the API key itself.
    fields = pyjson.dumps(dataclasses.astuple(settings))
    digest = hashlib.sha256(fields.encode()).hexdigest()
    client: TypeSafeClient = runtime.state(
        f"{_POOL_KEY}/{digest}", lambda: open_client(settings), close=lambda c: c.close()
    )
    return client


# --- Values across the boundary --------------------------------------------


def _unwrap(option: object) -> Any:
    """The payload of an AgL ``Option``, else ``None``."""
    return option.value if isinstance(option, Option.Some) else None


def _optional(value: object | None) -> object:
    return option_none() if value is None else option_some(value)


def _to_wire(payload: object) -> Any:
    """An AgL JSON payload as plain JSON for the SDK; decimals become floats."""
    return pyjson.loads(dumps_exact(payload, indent=None))


def _from_wire(value: object) -> object:
    """Plain SDK JSON as an AgL ``json`` value; non-integral numbers become decimals."""
    return agl_json(parse_json_strict(pyjson.dumps(value)))


def _decimal(value: float) -> Decimal:
    return Decimal(repr(value))


def _timeout_seconds(timeout: str | None) -> float | None:
    """Seconds of a duration text; any other text, zero, or too large raises ``TypeError``."""
    if timeout is None:
        return None
    try:
        return parse_positive_timeout(timeout)
    except ValueError as exc:
        raise AglException(TypeError(message=f"invalid timeout: {exc}")) from exc


# --- Wire questions --------------------------------------------------------


def _instruct(question: QuestionModel, instructions: JSONContent | None) -> None:
    """Give *question* its *instructions*; ``None`` stays off the wire, as SDK questions do."""
    if instructions is not None:
        question["instructions"] = instructions


def _noul_question(instructions: JSONContent | None, criteria: object) -> NoulModel:
    """A noul question from plain *instructions* and an AgL ``Option[NoulCriteria]``."""
    question: NoulModel = {"type": "noul"}
    _instruct(question, instructions)
    given = _unwrap(criteria)
    if given is not None:
        question["criteria"] = {
            "true": _to_wire(getattr(given, "when-true").value),
            "false": _to_wire(getattr(given, "when-false").value),
        }
    return question


def _choice_question(
    instructions: JSONContent | None, options: Mapping[str, JSONContent | None]
) -> ChoiceModel:
    """A choice question from plain *instructions* and label descriptions."""
    question: ChoiceModel = {"type": "choice", "criteria": options}
    _instruct(question, instructions)
    return question


def _score_question(instructions: JSONContent | None, levels: Sequence[JSONContent]) -> ScoreModel:
    """A score question from plain *instructions* and ordered level descriptions."""
    question: ScoreModel = {"type": "score", "criteria": levels}
    _instruct(question, instructions)
    return question


def _wire_question(question: Any) -> QuestionModel:
    """An AgL ``Question`` as a wire question."""
    instructions = _to_wire(question.instructions.value)
    if isinstance(question, _jev.Question.NoulQuestion):
        return _noul_question(instructions, question.criteria)
    if isinstance(question, _jev.Question.ChoiceQuestion):
        options = {label: _to_wire(text.value) for label, text in question.options.items()}
        return _choice_question(instructions, options)
    return _score_question(instructions, [_to_wire(level.value) for level in question.levels])


# --- Answers ---------------------------------------------------------------


def _graded(answer: ChoiceAnswer | ScoreAnswer) -> dict[str, object]:
    """The AgL ``confidence`` and ``probabilities`` of *answer*, and a score's ``score``."""
    if isinstance(answer, ChoiceAnswer):
        probabilities = {label: _decimal(p) for label, p in answer.probabilities.items()}
        return {"confidence": _decimal(answer.confidence), "probabilities": agl_dict(probabilities)}
    return {
        "score": _decimal(answer.score),
        "confidence": _decimal(answer.confidence),
        "probabilities": array([_decimal(p) for _, p in sorted(answer.probabilities.items())]),
    }


def _answer(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> object:
    """An SDK answer as an AgL ``Answer``."""
    if isinstance(answer, NoulAnswer):
        return _jev.Answer.NoulAnswer(noul=_decimal(answer.noul))
    if isinstance(answer, ChoiceAnswer):
        return _jev.Answer.ChoiceAnswer(choice=answer.choice, **_graded(answer))
    return _jev.Answer.ScoreAnswer(
        **_graded(answer),
        legend=array([_from_wire(text) for _, text in sorted(answer.legend.items())]),
    )


def _request_id(response: SystemOneResponse) -> str | None:
    return response.raw_http_response.headers.get(_REQUEST_ID_HEADER)


def _answer_named[A](response: SystemOneResponse, name: str, kind: type[A]) -> A:
    """The answer to question *name*; a missing or mistyped one raises ``JevResponseError``."""
    answer = response.answers.get(name)
    if not isinstance(answer, kind):
        message = f"The response has no {kind.__name__} for question {name!r}."
        raise _invalid_response(response, f"answers.{name}", message)
    return answer


def _invalid_response(response: SystemOneResponse, field_path: str, message: str) -> AglException:
    """A traced ``JevResponseError`` at *field_path* of *response*."""
    return _jev_error(
        "JevResponseError", message, _request_id(response), {"field-path": field_path}
    )


def _response(response: SystemOneResponse) -> object:
    """An SDK response as an AgL ``Response``."""
    usage = _jev.Usage(
        **{
            "input-tokens": _optional(response.usage.input_tokens),
            "output-tokens": _optional(response.usage.output_tokens),
        }
    )
    answers = {name: _answer(answer) for name, answer in response.answers.items()}
    return _jev.Response(
        model=response.model,
        answers=agl_dict(answers),
        usage=usage,
        **{"request-id": _optional(_request_id(response))},
    )


# --- Errors ----------------------------------------------------------------


def _error_body(body: object) -> object:
    """An SDK error body as AgL ``json``; one ``json`` cannot hold (such as ``NaN``) is ``null``."""
    try:
        return _from_wire(body)
    except StrictJsonParseError:
        return agl_json(None)


def _api_fields(error: Any) -> dict[str, object]:
    return {"status": error.status, "body": _error_body(error.body)}


def _rate_limit_fields(error: Any) -> dict[str, object]:
    delay = error.retry_after_ms
    retry_after = None if delay is None else round(delay)
    return {**_api_fields(error), "retry-after-ms": _optional(retry_after)}


def _timeout_fields(error: Any) -> dict[str, object]:
    return {"timeout": format_timeout(error.timeout)}


def _connection_fields(error: Any) -> dict[str, object]:
    return {"cause": str(error.__cause__ or error)}


def _no_fields(error: Any) -> dict[str, object]:
    return {}


# SDK class -> (AgL exception name, its own fields). First match wins, so a
# subclass precedes its base.
_ERRORS: tuple[tuple[type[TypeSafeError], str, Callable[[Any], dict[str, object]]], ...] = (
    (
        TypeSafeAPIResponseValidationError,
        "JevResponseError",
        lambda error: {"field-path": error.field_path},
    ),
    (TypeSafeAuthenticationError, "JevAuthError", _api_fields),
    (TypeSafePermissionDeniedError, "JevAuthError", _api_fields),
    (TypeSafeBadRequestError, "JevRequestError", _api_fields),
    (TypeSafeNotFoundError, "JevRequestError", _api_fields),
    (TypeSafeUnprocessableEntityError, "JevRequestError", _api_fields),
    (TypeSafeRateLimitError, "JevRateLimitError", _rate_limit_fields),
    (TypeSafeInternalServerError, "JevServerError", _api_fields),
    (TypeSafeAPIError, "JevApiError", _api_fields),
    (TypeSafeAPITimeoutError, "JevTimeoutError", _timeout_fields),
    (TypeSafeAPIConnectionError, "JevConnectionError", _connection_fields),
    (TypeSafeError, "JevError", _no_fields),
)


def _jev_error(
    name: str, message: str, request_id: str | None, fields: Mapping[str, object]
) -> AglException:
    """Trace a ``jev_failure`` record and build the ``JevError``-family exception *name*.

    *fields* are the exception's own fields; a ``status`` among them is traced.
    """
    runtime.trace(
        "jev_failure",
        {"error_type": name, "status": fields.get("status"), "request_id": request_id},
    )
    exception = getattr(_jev, name)
    return AglException(
        exception(message=message, **{"request-id": _optional(request_id)}, **fields)
    )


def _failure(error: TypeSafeError) -> AglException:
    """*error* as its traced AgL exception."""
    name, fields = next((name, fields) for cls, name, fields in _ERRORS if isinstance(error, cls))
    request_id = error.request_id if isinstance(error, TypeSafeAPIError) else None
    return _jev_error(name, str(error), request_id, fields(error))


# --- Requests --------------------------------------------------------------


def _send(
    state: Any,
    questions: Mapping[str, QuestionModel],
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> SystemOneResponse:
    """Ask wire *questions* about the AgL json *state*; the rest are the extern's settings."""
    model_name = _unwrap(model)
    seconds = _timeout_seconds(_unwrap(timeout))
    wire_state = _to_wire(state.value)
    runtime.trace("jev_request", {"model": model_name, "state": wire_state, "questions": questions})
    settings = Settings(_unwrap(api_key), _unwrap(base_url), max_retries)
    try:
        response = _client(settings).system_one(
            wire_state, questions, model=model_name, timeout=seconds
        )
    except TypeSafeError as error:
        raise _failure(error) from error
    runtime.trace(
        "jev_response",
        {
            "model": response.model,
            "answers": {
                name: answer.model_dump(mode="json") for name, answer in response.answers.items()
            },
            "usage": response.usage.model_dump(mode="json"),
            "request_id": _request_id(response),
        },
    )
    return response


# --- Typed targets ---------------------------------------------------------


class _Unsupported(Exception):
    """A target type no question kind maps, and why; *target* locates it when it is not the
    call's target (an ``ask-many`` field)."""

    def __init__(self, why: str, target: str | None = None) -> None:
        super().__init__(why)
        self.target = target


class _Outside(Exception):
    """An answer field outside the target type; the argument is the field's name."""


@dataclass(frozen=True)
class _Typed:
    """The question a target type asks and the builder of a target value from its answer.

    ``value`` raises ``_Outside`` for an answer the target cannot hold.
    """

    question: Callable[[JSONContent | None], QuestionModel]
    answer: type[NoulAnswer] | type[ChoiceAnswer] | type[ScoreAnswer]
    value: Callable[[Any], object]


def _classify[R](target: TypeContract, classify: Callable[[TypeContract], R]) -> R:
    """``classify(target)``; an unsupported target raises a traced ``JevTargetError``."""
    try:
        return classify(target)
    except _Unsupported as why:
        located = why.target or target.label
        message = f"{located} is not a Jev target: {why}"
        raise _jev_error("JevTargetError", message, None, {"target": located}) from None


def _build(contract: TypeContract, **fields: object) -> object:
    """A value of *contract*'s record or member class from its declared-name *fields*."""
    nominal: Any = contract.nominal
    return nominal(**fields)


def _typed(target: TypeContract, threshold: Decimal) -> _Typed:
    """The question *target* selects; ``bool`` is the noul probability reaching *threshold*."""
    if target.kind == "bool":
        return _noul(lambda probability: probability >= threshold)
    if target.nominal is _jev.Noul:
        return _noul(_noul_record)
    if target.nominal is _jev.Choice:
        return _choice(target.fields["choice"].contract, wrap=True)
    if target.nominal is _jev.Score:
        return _score(target.fields["level"].contract)
    if target.kind == "enum":
        return _choice(target, wrap=False)
    raise _Unsupported("not bool, Noul, a fieldless enum, Choice, or Score")


def _noul(value: Callable[[Decimal], object], criteria: object = _NO_CRITERIA) -> _Typed:
    """A noul question with AgL ``Option[NoulCriteria]`` *criteria*; *value* maps the
    probability."""
    return _Typed(
        lambda instructions: _noul_question(instructions, criteria),
        NoulAnswer,
        lambda answer: value(_decimal(answer.noul)),
    )


def _noul_record(probability: Decimal) -> object:
    return _jev.Noul(probability=probability)


def _members(enum: TypeContract) -> Mapping[str, TypeContract]:
    """*enum*'s members by tag, when it has at least two, all fieldless."""
    if enum.kind != "enum":
        raise _Unsupported(f"{enum.label} is not an enum")
    if len(enum.members) < 2:
        raise _Unsupported(f"{enum.label} has fewer than two members")
    members: Mapping[str, TypeContract] = enum.members
    for tag, member in members.items():
        if member.fields:
            raise _Unsupported(f"member {tag!r} of {enum.label} has fields")
    return members


def _choice(enum: TypeContract, *, wrap: bool) -> _Typed:
    """A choice over *enum*'s members; *wrap* answers a ``Choice`` rather than the member."""
    members = _members(enum)
    options = {tag: member.doc for tag, member in members.items()}

    def value(answer: ChoiceAnswer) -> object:
        member = members.get(answer.choice)
        if member is None:
            raise _Outside("choice")
        # The API gives every label a probability.
        if answer.probabilities.keys() != members.keys():
            raise _Outside("probabilities")
        chosen = _build(member)
        return _jev.Choice(choice=chosen, **_graded(answer)) if wrap else chosen

    return _Typed(lambda instructions: _choice_question(instructions, options), ChoiceAnswer, value)


def _score(enum: TypeContract) -> _Typed:
    """A score whose rubric is *enum*'s members in declaration order."""
    members = _members(enum)
    if len(members) > _MAX_LEVELS:
        raise _Unsupported(f"{enum.label} has more than {_MAX_LEVELS} levels")
    levels = [member.doc or tag for tag, member in members.items()]
    rubric = list(members.values())

    def value(answer: ScoreAnswer) -> object:
        # The API gives every level a probability.
        indices = range(len(rubric))
        if answer.probabilities.keys() != set(indices):
            raise _Outside("probabilities")
        # The most probable level; `max` keeps the first, so the lowest index on a tie.
        best = max(indices, key=answer.probabilities.__getitem__)
        return _jev.Score(level=_build(rubric[best]), **_graded(answer))

    return _Typed(lambda instructions: _score_question(instructions, levels), ScoreAnswer, value)


def _record_fields(
    record: TypeContract, threshold: Decimal
) -> tuple[dict[str, tuple[_Typed, str]], dict[str, str]]:
    """*record*'s typed questions with their instructions (the field's ``@doc``, else its
    name), and its declared field names, both by field JSON name."""
    if record.kind != "record":
        raise _Unsupported("ask-many needs a record")
    if not record.fields:
        raise _Unsupported("ask-many needs at least one field")
    questions = {}
    for key, field_ in record.fields.items():
        try:
            typed = _typed(field_.contract, threshold)
        except _Unsupported as why:
            raise _Unsupported(str(why), f"{record.label}.{field_.name}") from None
        questions[key] = (typed, field_.doc or field_.name)
    names = {key: field_.name for key, field_ in record.fields.items()}
    return questions, names


def _ask_typed(
    state: object,
    questions: Mapping[str, tuple[_Typed, JSONContent | None]],
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> dict[str, object]:
    """Ask typed *questions*, each with its instructions, in one request; target values by
    question name."""
    wire = {name: typed.question(instructions) for name, (typed, instructions) in questions.items()}
    response = _send(state, wire, model, timeout, api_key, base_url, max_retries)
    values = {}
    for name, (typed, _) in questions.items():
        answer = _answer_named(response, name, typed.answer)
        try:
            values[name] = typed.value(answer)
        except _Outside as outside:
            path = f"answers.{name}.{outside}"
            raise _invalid_response(response, path, f"{path} is outside the target") from None
    return values


def _ask_one(
    typed: _Typed,
    instructions: str,
    state: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> object:
    """Ask one typed question; its target value."""
    questions = {_QUESTION: (typed, instructions)}
    values = _ask_typed(state, questions, model, timeout, api_key, base_url, max_retries)
    return values[_QUESTION]


def system_one(
    state: object,
    questions: Mapping[str, object],
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> object:
    """Ask *questions* about *state* in one request."""
    wire = {name: _wire_question(question) for name, question in questions.items()}
    return _response(_send(state, wire, model, timeout, api_key, base_url, max_retries))


def ask_noul(
    instructions: str,
    state: object,
    criteria: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> object:
    """Ask one noul question."""
    typed = _noul(_noul_record, criteria)
    return _ask_one(typed, instructions, state, model, timeout, api_key, base_url, max_retries)


def ask_choice(
    target: TypeContract,
    instructions: str,
    state: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> object:
    """Ask one choice question over the members of *target*'s choice type."""
    typed = _classify(target, lambda enum: _choice(enum, wrap=True))
    return _ask_one(typed, instructions, state, model, timeout, api_key, base_url, max_retries)


def ask_score(
    target: TypeContract,
    instructions: str,
    state: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
) -> object:
    """Ask one score question over the rubric of *target*'s level type."""
    typed = _classify(target, _score)
    return _ask_one(typed, instructions, state, model, timeout, api_key, base_url, max_retries)


def ask(
    target: TypeContract,
    instructions: str,
    state: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
    noul_threshold: Decimal,
) -> object:
    """Ask one question whose kind *target* selects."""
    typed = _classify(target, lambda contract: _typed(contract, noul_threshold))
    return _ask_one(typed, instructions, state, model, timeout, api_key, base_url, max_retries)


def ask_many(
    target: TypeContract,
    state: object,
    model: object,
    timeout: object,
    api_key: object,
    base_url: object,
    max_retries: int,
    noul_threshold: Decimal,
) -> object:
    """Ask one question per field of the record *target* in one request."""
    questions, names = _classify(target, lambda record: _record_fields(record, noul_threshold))
    values = _ask_typed(state, questions, model, timeout, api_key, base_url, max_retries)
    return _build(target, **{names[key]: value for key, value in values.items()})

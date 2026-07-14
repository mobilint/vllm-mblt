import json

import pytest

from tests.repro_openai_echo_logprobs_concurrency import Response, validate_echo_logprobs_response


def make_response(*, text: str = "Hello world!") -> Response:
    return Response(
        status=200,
        body=json.dumps(
            {
                "choices": [
                    {
                        "text": text,
                        "logprobs": {
                            "tokens": ["Hello", " world", "!"],
                            "token_logprobs": [None, -0.2, -0.3],
                            "top_logprobs": [None, {" world": -0.2}, {"!": -0.3}],
                            "text_offset": [0, 5, 11],
                        },
                    }
                ]
            }
        ),
    )


def test_validate_echo_logprobs_response_accepts_aligned_prompt_logprobs() -> None:
    validate_echo_logprobs_response(make_response(), "Hello world!", name="test")


@pytest.mark.parametrize("status", [400, 501])
def test_validate_echo_logprobs_response_rejects_unsupported_statuses(status: int) -> None:
    with pytest.raises(RuntimeError, match="expected HTTP 200"):
        validate_echo_logprobs_response(Response(status=status, body="{}"), "Hello world!", name="test")


def test_validate_echo_logprobs_response_rejects_missing_logprobs() -> None:
    response = Response(status=200, body=json.dumps({"choices": [{"text": "Hello world!"}]}))

    with pytest.raises(RuntimeError, match="logprobs"):
        validate_echo_logprobs_response(response, "Hello world!", name="test")


def test_validate_echo_logprobs_response_rejects_misaligned_prompt_tokens() -> None:
    response = make_response(text="Hello world!")
    payload = json.loads(response.body)
    payload["choices"][0]["logprobs"]["tokens"] = ["Hello", " there", "!"]
    response = Response(status=200, body=json.dumps(payload))

    with pytest.raises(RuntimeError, match="diverges from prompt"):
        validate_echo_logprobs_response(response, "Hello world!", name="test")


def test_validate_echo_logprobs_response_rejects_nonfinite_prompt_logprob() -> None:
    response = make_response()
    payload = json.loads(response.body)
    payload["choices"][0]["logprobs"]["token_logprobs"][1] = float("inf")
    response = Response(status=200, body=json.dumps(payload))

    with pytest.raises(RuntimeError, match="not finite"):
        validate_echo_logprobs_response(response, "Hello world!", name="test")


def test_validate_echo_logprobs_response_rejects_extra_null_token_logprob() -> None:
    response = make_response()
    payload = json.loads(response.body)
    payload["choices"][0]["logprobs"]["token_logprobs"][2] = None
    response = Response(status=200, body=json.dumps(payload))

    with pytest.raises(RuntimeError, match="not finite"):
        validate_echo_logprobs_response(response, "Hello world!", name="test")

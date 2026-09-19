import json

import pytest
from mwm_harness.streaming import StreamAssembler, parse_sse_line


def _chunk(delta=None, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]}


def _fragment(index, arguments="", call_id=None, name=None):
    fragment = {"index": index, "function": {"arguments": arguments}}
    if call_id:
        fragment["id"] = call_id
    if name:
        fragment["function"]["name"] = name
    return {"tool_calls": [fragment]}


def _run(chunks):
    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    return assembler.finish()


def test_text_and_reasoning_are_joined_separately():
    turn = _run(
        [
            _chunk({"reasoning_content": "think "}),
            _chunk({"reasoning_content": "more"}),
            _chunk({"content": "Hel"}),
            _chunk({"content": "lo"}, finish_reason="stop"),
        ]
    )
    assert turn.text == "Hello"
    assert turn.reasoning == "think more"
    assert turn.tool_calls == []
    assert turn.finish_reason == "stop"


def test_interleaved_parallel_calls_are_reassembled_by_index():
    turn = _run(
        [
            _chunk(_fragment(0, '{"ci', call_id="call_a", name="get_weather")),
            _chunk(_fragment(1, '{"zone"', call_id="call_b", name="get_time")),
            _chunk(_fragment(0, 'ty": "Oslo"}')),
            _chunk(_fragment(1, ': "Europe/Oslo"}')),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("call_a", "get_weather", {"city": "Oslo"}),
        ("call_b", "get_time", {"zone": "Europe/Oslo"}),
    ]
    assert all(c.error is None for c in turn.tool_calls)
    assert turn.finish_reason == "tool_calls"


def test_id_only_in_first_fragment_is_kept():
    turn = _run(
        [
            _chunk(_fragment(0, "", call_id="call_x", name="get_time")),
            _chunk(_fragment(0, '{"zone": "UTC"}')),
        ]
    )
    assert turn.tool_calls[0].id == "call_x"
    assert turn.tool_calls[0].arguments == {"zone": "UTC"}


def test_malformed_arguments_report_an_error_instead_of_raising():
    turn = _run([_chunk(_fragment(0, '{"city": "Os', call_id="call_a", name="get_weather"))])
    call = turn.tool_calls[0]
    assert call.arguments is None
    assert "not valid JSON" in call.error
    assert call.raw_arguments == '{"city": "Os'


def test_non_object_arguments_are_rejected():
    turn = _run([_chunk(_fragment(0, "[1, 2]", call_id="call_a", name="get_weather"))])
    assert turn.tool_calls[0].arguments is None
    assert "JSON object" in turn.tool_calls[0].error


def test_empty_arguments_mean_no_arguments():
    turn = _run([_chunk(_fragment(0, "", call_id="call_a", name="get_time"))])
    assert turn.tool_calls[0].arguments == {}
    assert turn.tool_calls[0].error is None


def test_missing_function_name_is_an_error():
    turn = _run([_chunk(_fragment(0, "{}", call_id="call_a"))])
    assert turn.tool_calls[0].error == "tool call has no function name"


def test_usage_arrives_in_a_final_chunk_with_no_choices():
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 100},
    }
    turn = _run([_chunk({"content": "OK"}, finish_reason="stop"), {"choices": [], "usage": usage}])
    assert turn.usage == usage
    assert turn.cached_tokens == 100


def test_cached_tokens_is_none_when_the_provider_does_not_report_it():
    turn = _run([{"choices": [], "usage": {"prompt_tokens": 5}}])
    assert turn.cached_tokens is None
    assert _run([]).cached_tokens is None


@pytest.mark.parametrize("line", ["", ": keep-alive", "event: ping", "data: [DONE]", "data:"])
def test_parse_sse_line_skips_non_payload_lines(line):
    assert parse_sse_line(line) is None


def test_parse_sse_line_reads_a_data_line():
    chunk = _chunk({"content": "hi"})
    assert parse_sse_line("data: " + json.dumps(chunk)) == chunk

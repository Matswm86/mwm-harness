"""Vision: a tool result can carry an image, and only a vision model is sent it.

The wire shape is measured, not assumed: ``spike/spike_vision.py`` sends gold-set
frames to the pay-as-you-go models and COMPAT.md records what came back.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from conftest import MODEL
from mwm_harness.mcp_client import render_result
from mwm_harness.messages import (
    MAX_IMAGE_BYTES,
    Message,
    encode_image,
    image_block,
    image_problem,
    text_block,
    to_openai,
    tool_result_block,
    tool_use_block,
)
from mwm_harness.observation_pack import pack_observations
from mwm_harness.providers import chunks_for
from mwm_harness.tools.base import ToolContext
from mwm_harness.tools.files import Read


def run(coro):
    return asyncio.run(coro)


PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
VISION_MODEL = replace(MODEL, id="vision-model", vision=True)


def png(tmp_path: Path, name: str = "frame.png", raw: bytes = PIXEL) -> Path:
    path = tmp_path / name
    path.write_bytes(raw)
    return path


def ctx(tmp_path: Path) -> ToolContext:
    from mwm_harness.sandbox import Sandbox

    return ToolContext(cwd=tmp_path, scratch=tmp_path / "scratch", sandbox=Sandbox("off", []))


def turn_with(call_id: str, result: dict[str, Any]) -> list[Message]:
    """A history where one tool call has already been answered with ``result``."""
    return [
        Message("user", "what does the chart show"),
        Message("assistant", [tool_use_block("Read", {"file_path": "frame.png"}, call_id)]),
        Message("user", [result]),
    ]


# --------------------------------------------------------------------- blocks


def test_image_problem_accepts_a_real_image_and_names_the_rest(tmp_path: Path):
    assert image_problem(png(tmp_path)) is None
    text = tmp_path / "notes.md"
    text.write_text("hi", encoding="utf-8")
    assert "not an image" in (image_problem(text) or "")
    assert "file not found" in (image_problem(tmp_path / "gone.png") or "")


def test_image_problem_refuses_an_oversize_file(tmp_path: Path):
    big = png(tmp_path, "huge.png", b"\x89PNG" + b"0" * (MAX_IMAGE_BYTES + 1))
    assert "over the" in (image_problem(big) or "")


def test_image_block_holds_a_path_and_no_bytes(tmp_path: Path):
    path = png(tmp_path)
    block = image_block(path)
    assert block["type"] == "image"
    assert block["source"] == {
        "type": "file",
        "path": str(path),
        "media_type": "image/png",
    }
    assert len(json.dumps(block)) < 200  # the transcript stays small


def test_encode_image_reads_the_file_and_reports_a_missing_one(tmp_path: Path):
    url, note = encode_image(image_block(png(tmp_path)))
    assert url is not None and url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PIXEL
    assert note == ""
    gone, note = encode_image(image_block(tmp_path / "gone.png"))
    assert gone is None and "could not be read" in note


def test_result_with_an_image_keeps_its_text_readable(tmp_path: Path):
    block = tool_result_block("toolu_1", "[image frame.png]", images=[image_block(png(tmp_path))])
    assert isinstance(block["content"], list)
    assert block["content"][0] == {"type": "text", "text": "[image frame.png]"}
    assert block["content"][1]["type"] == "image"
    assert tool_result_block("toolu_2", "plain")["content"] == "plain"


# ------------------------------------------------------------------- the wire


def test_image_goes_out_as_a_user_part_after_the_tool_message(tmp_path: Path):
    path = png(tmp_path)
    history = turn_with(
        "toolu_1", tool_result_block("toolu_1", f"[image {path}]", images=[image_block(path)])
    )
    out = to_openai("system", history, vision=True)
    assert [entry["role"] for entry in out] == ["system", "user", "assistant", "tool", "user"]
    tool_message, image_message = out[3], out[4]
    assert tool_message["tool_call_id"] == "toolu_1"
    assert isinstance(tool_message["content"], str)  # a tool message carries no image
    parts = image_message["content"]
    assert parts[0]["type"] == "image_url"
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == PIXEL
    # The result text stays in the tool message; the image turn says where it came from.
    assert tool_message["content"] == f"[image {path}]"
    assert parts[1]["type"] == "text" and str(path) in parts[1]["text"]


def test_a_model_without_vision_gets_the_path_instead_of_the_bytes(tmp_path: Path):
    path = png(tmp_path)
    history = turn_with(
        "toolu_1", tool_result_block("toolu_1", "[image]", images=[image_block(path)])
    )
    out = to_openai("", history, vision=False)
    assert "image_url" not in json.dumps(out)
    note = out[-1]["content"]
    assert isinstance(note, str) and str(path) in note and "text only" in note


def test_an_image_that_vanished_becomes_a_note_not_a_crash(tmp_path: Path):
    history = turn_with(
        "toolu_1",
        tool_result_block("toolu_1", "[image]", images=[image_block(tmp_path / "gone.png")]),
    )
    out = to_openai("", history, vision=True)
    assert "image_url" not in json.dumps(out)
    assert "could not be read" in out[-1]["content"]


def test_two_results_in_one_turn_share_one_image_message(tmp_path: Path):
    first, second = png(tmp_path, "a.png"), png(tmp_path, "b.png")
    history = [
        Message("assistant", [tool_use_block("Read", {}, "t1"), tool_use_block("Read", {}, "t2")]),
        Message(
            "user",
            [
                tool_result_block("t1", "a", images=[image_block(first)]),
                tool_result_block("t2", "b", images=[image_block(second)]),
            ],
        ),
    ]
    out = to_openai("", history, vision=True)
    assert [entry["role"] for entry in out] == ["assistant", "tool", "tool", "user"]
    assert sum(p["type"] == "image_url" for p in out[-1]["content"]) == 2


def test_two_user_turns_with_images_stay_one_message(tmp_path: Path):
    path = png(tmp_path)
    image = image_block(path)
    history = [
        Message("user", [tool_result_block("t1", "a", images=[image])]),
        Message("user", "and again", is_meta=True),
    ]
    out = to_openai("", history, vision=True)
    # Some endpoints reject two user turns in a row, so the text joins the image message.
    assert [entry["role"] for entry in out] == ["tool", "user"]
    assert out[-1]["content"][-1] == {"type": "text", "text": "and again"}


# ----------------------------------------------------------------- Read tool


def test_read_returns_the_picture_for_an_image(tmp_path: Path):
    path = png(tmp_path)
    result = _run(Read(), {"file_path": str(path)}, tmp_path)
    assert not result.is_error
    assert str(path) in result.content and "image/png" in result.content
    assert result.images == [image_block(path, "image/png")]


def test_read_still_numbers_a_text_file(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("first\nsecond\n", encoding="utf-8")
    result = _run(Read(), {"file_path": str(path)}, tmp_path)
    assert result.images == []
    assert "1\tfirst" in result.content


def test_read_refuses_an_image_over_the_limit(tmp_path: Path):
    path = png(tmp_path, "huge.png", b"\x89PNG" + b"0" * (MAX_IMAGE_BYTES + 1))
    result = _run(Read(), {"file_path": str(path)}, tmp_path)
    assert result.is_error and result.images == []


def _run(tool: Any, tool_input: dict[str, Any], tmp_path: Path):
    return run(tool.run(tool_input, ctx(tmp_path)))


# ------------------------------------------------------------------- mcp side


def test_mcp_image_is_saved_and_offered_to_the_model(tmp_path: Path):
    result = {
        "content": [
            {"type": "text", "text": "here is the frame"},
            {"type": "image", "mimeType": "image/png", "data": base64.b64encode(PIXEL).decode()},
        ]
    }
    text, images = render_result(result, tmp_path)
    assert "here is the frame" in text and "saved to" in text
    saved = Path(images[0]["source"]["path"])
    assert saved.read_bytes() == PIXEL and saved.parent == tmp_path
    assert text.count(str(saved)) == 1  # the path stays in the text for a text-only model


def test_mcp_audio_is_saved_without_an_image_block(tmp_path: Path):
    result = {"content": [{"type": "audio", "mimeType": "audio/wav", "data": "UklGRg=="}]}
    text, images = render_result(result, tmp_path)
    assert images == [] and "saved to" in text


# ---------------------------------------------------------------- cost over time


def test_an_old_result_keeps_the_path_and_loses_the_image(tmp_path: Path):
    path = png(tmp_path)

    def history(assistants_after: int) -> list[Message]:
        messages = turn_with(
            "toolu_1", tool_result_block("toolu_1", "[image]", images=[image_block(path)])
        )
        return messages + [
            Message("assistant", [text_block(f"step {i}")]) for i in range(assistants_after)
        ]

    for after in (0, 1):
        packed, _ = pack_observations(history(after), tmp_path / f"scratch{after}")
        assert packed[2].content[0]["content"][1]["type"] == "image"
    packed, _ = pack_observations(history(2), tmp_path / "scratch2")
    content = packed[2].content[0]["content"]
    assert isinstance(content, str) and str(path) in content and "left out" in content


# ----------------------------------------------------------------- end to end


def test_a_read_of_an_image_reaches_the_next_request(make_session):
    session, _, provider = make_session(
        [
            chunks_for(tool_calls=[("Read", {"file_path": "frame.png"})]),
            chunks_for("the chart shows one red pixel"),
        ]
    )
    session.set_model(VISION_MODEL)
    png(session.cwd)
    ended = run(session.send("look at the frame"))
    assert ended.text == "the chart shows one red pixel"
    parts = [
        part
        for entry in provider.requests[1]["messages"]
        for part in _parts(entry)
        if part["type"] == "image_url"
    ]
    assert len(parts) == 1
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == PIXEL
    # The stored history names the file; only the request carries the bytes.
    stored = json.dumps([message.content for message in session.messages], default=str)
    assert base64.b64encode(PIXEL).decode() not in stored


def test_the_same_turn_on_a_text_model_sends_no_image(make_session):
    session, _, provider = make_session(
        [
            chunks_for(tool_calls=[("Read", {"file_path": "frame.png"})]),
            chunks_for("I cannot see it."),
        ]
    )
    png(session.cwd)
    run(session.send("look at the frame"))
    assert "image_url" not in json.dumps(provider.requests[1]["messages"])
    assert "text only" in json.dumps(provider.requests[1]["messages"])


def _parts(entry: dict[str, Any]) -> list[dict[str, Any]]:
    content = entry.get("content")
    return content if isinstance(content, list) else []

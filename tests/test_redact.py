"""Secret stripping. The first block ports Shrike's test/redact.test.ts (MIT) case for case."""

from __future__ import annotations

import json

import pytest
from mwm_harness import redact
from mwm_harness.redact import Redactor, count_labels, entropy, redact_text, secret_literals

# ---------------------------------------------------------------- Shrike's cases


def test_keys_inside_json_keep_the_key_and_stay_valid_json():
    out = redact_text(
        '{"email":"a@b.co","password":"hunter2","nested":{"apiKey":"abc123",'
        '"card":4242424242424242,"author":"Ann"},"maxTokens":100}'
    )
    assert json.loads(out) == {
        "email": "a@b.co",
        "password": "[redacted:password]",
        "nested": {"apiKey": "[redacted:apiKey]", "card": "[redacted:card]", "author": "Ann"},
        "maxTokens": 100,
    }
    assert (
        redact_text('{ "access_token" : "x.y.z" }')
        == '{ "access_token" : "[redacted:access_token]" }'
    )
    assert redact_text('{"password":""}') == '{"password":""}'


def test_keys_inside_inspect_style_strings():
    assert redact_text(
        "{ title: 'buy milk', password: 'hunter2', 'api-key': 'k1', sessionId: \"s's\", authorId: 3, ok: true }"
    ) == (
        "{ title: 'buy milk', password: '[redacted:password]', 'api-key': '[redacted:api-key]', "
        'sessionId: "[redacted:sessionId]", authorId: 3, ok: true }'
    )
    keep = "req { POST /login body: { user: 'ann', passwd: '[redacted]' } }"
    assert redact_text(keep) == keep
    keep = "{ token: [Object], session: { id: 1 }, authenticated: false }"
    assert redact_text(keep) == keep


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("GET /api/x?token=abc123&page=2", "GET /api/x?token=[redacted:token]&page=2"),
        ("user=ann&password=p%40ss&remember=1", "user=ann&password=[redacted:password]&remember=1"),
        ("sessionid=abc123; theme=dark", "sessionid=[redacted:sessionid]; theme=dark"),
        ("sent Bearer abcdefghijkl to api", "sent Bearer [redacted:token] to api"),
        (
            "UPDATE users SET password = 'x1' WHERE id = ?",
            "UPDATE users SET password = '[redacted:password]' WHERE id = ?",
        ),
        (
            "SELECT * FROM sessions WHERE token = ? AND user_id = 4",
            "SELECT * FROM sessions WHERE token = ? AND user_id = 4",
        ),
    ],
)
def test_query_strings_form_data_cookies_bearer(text, want):
    assert redact_text(text) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("key AKIAIOSFODNN7EXAMPLE here", "key [redacted:aws_key] here"),
        ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345", "[redacted:anthropic_key]"),
        ("using sk-proj-abcdefghijklmnopqrstuvwx", "using [redacted:openai_key]"),
        ("sk_live_abcdefghij1234", "[redacted:stripe_key]"),
        ("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "[redacted:github_token]"),
        ("xoxb-1234567890-abcdefghij", "[redacted:slack_token]"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcdefSIGNATURE_x", "[redacted:jwt]"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
            "[redacted:pem]",
        ),
        (
            "postgres://admin:s3cret@db.local:5432/app",
            "postgres://admin:[redacted:password]@db.local:5432/app",
        ),
        ("mongodb+srv://u:p@x.net", "mongodb+srv://u:[redacted:password]@x.net"),
        # added for the harness's own providers
        ("GROQ gsk_abcdefghijklmnopqrstuvwxyz0123", "GROQ [redacted:groq_key]"),
        (
            "hf_" + "abcdefghijklmnopqrstuvwxyzABCDEFGH",
            "[redacted:huggingface_token]",
        ),  # split: push protection
        ("dashscope sk-0123456789abcdef0123456789abcdef", "dashscope [redacted:openai_key]"),
    ],
)
def test_known_secret_shapes(text, want):
    assert redact_text(text) == want


def test_high_entropy_tokens_in_value_positions():
    assert redact_text("{ id: 'q8Zr3LmX0vT7pN2kWc5yB9dHs4' }") == "{ id: '[redacted:secret]' }"
    assert redact_text("X-Custom: q8Zr3LmX0vT7pN2kWc5yB9dHs4") == "X-Custom: [redacted:secret]"
    assert entropy("q8Zr3LmX0vT7pN2kWc5yB9dHs4") > 4
    assert (
        redact_text("verifyToken('9f86d081884c7d659a2feaa0c55ad015')")
        == "verifyToken('[redacted:secret]')"
    )
    uuid = "'550e8400-e29b-41d4-a716-446655440000'"
    assert redact_text(uuid) == uuid


@pytest.mark.parametrize(
    "text",
    [
        "NOT NULL constraint failed: todos.user_id",
        "INSERT INTO todos (title, user_id) VALUES (?, ?)",
        "{ title: 'buy milk', userId: undefined }",
        "creating todo { title: 'buy milk' }",
        "trace 0af7651916cd43dd8448eb211c80319c event b7ad6b7169203331",
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "commit 3f786850e387550fdab836ed7e6dc881de23001b",
        "/Users/me/Desktop/new-tool/packages/correlator/src/summary.ts",
        "at createTodo (http://localhost:5173/src/api.ts?t=1712345678901:12:21)",
        "https://example.com/api/v1/items/aB3dE5fG7hJ9kL1mN3pQ5rS7tU9?page=2",
        "'internationalizationFrameworkConfiguration'",
        "id: 550e8400-e29b-41d4-a716-446655440000",
        "Error: Cannot read properties of undefined (reading 'map')",
        "{ cards: [ 'a' ], passed: true, tokens: 12 }",
        "",
        # harness additions: ordinary tool output that must survive
        "/home/user/project/mwm_harness/loop.py:627",
        "max_tokens = 4096",
        "author = 'Ann'",
        "def tokenize(text: str) -> list[str]:",
        "git log: 7f5ad14 Emergency stop and undo-a-turn",
        "session_id: None",
    ],
)
def test_leaves_normal_text_alone(text):
    assert redact_text(text) == text


def test_idempotent():
    s = (
        '{"password":"x","q":"?token=abc","k":"sk-ant-abcdefghijklmnopqrstuvwxyz",'
        '"db":"postgres://a:b@h"} Bearer abcdefghijkl'
    )
    once = redact_text(s)
    assert redact_text(once) == once


def test_a_url_port_and_path_are_not_a_secret():
    url = "POST http://127.0.0.1:18080/google.firestore.v1.Firestore/Write/channel"
    assert redact_text(url) == url
    assert redact_text("note: sk9Qx2Lm4Np7Rt1Vw3Yz5Ab8Cd0Ef6Gh") == "note: [redacted:secret]"


def test_a_class_name_is_not_a_social_security_number():
    keep = "{ className: 'inline-flex items-center', variant: undefined }"
    assert redact_text(keep) == keep
    assert redact_text("{ userSsn: '123-45-6789' }") == "{ userSsn: '[redacted:userSsn]' }"


# ---------------------------------------------------------------- harness additions


def test_env_file_lines_lose_their_secret_values():
    text = "OLLAMA_API_KEY=abc123def456\nDB_HOST=localhost\nDEBUG=true\n"
    out = redact_text(text)
    assert "abc123def456" not in out and "OLLAMA_API_KEY=[redacted:OLLAMA_API_KEY]" in out
    assert "DB_HOST=localhost" in out and "DEBUG=true" in out


def test_literals_from_env_files_are_masked_anywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "workspace_root", lambda: tmp_path / "ws")
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=plainlowercasekeyvalue\nDB_HOST=localhost\nPORT=5432\nSHORT_TOKEN=abc\n"
    )
    project = tmp_path / "ws" / "projects" / "app"
    project.mkdir(parents=True)
    (project / ".env").write_text("export GATEWAY_PASSWORD='hunter2hunter2'\n")
    literals = secret_literals(tmp_path, secrets={"CLOUDFLARE_API_TOKEN": "cf-token-value-123"})
    names = {name for name, _ in literals}
    assert names == {"typesafe_api_key", "gateway_password", "cloudflare_api_token"}
    out = redact_text(
        "echo plainlowercasekeyvalue; curl -u x:hunter2hunter2 h; cf-token-value-123 at localhost:5432",
        literals,
    )
    assert "plainlowercasekeyvalue" not in out and "hunter2hunter2" not in out
    assert "cf-token-value-123" not in out and "localhost:5432" in out


def test_redactor_counts_new_labels_only():
    redactor = Redactor(redact.Path("."), literals=[])
    text, counts = redactor(
        "already [redacted:token] and password='hunter2' and sk-ant-" + "a" * 30
    )
    assert counts == {"password": 1, "anthropic_key": 1}
    assert redactor.total == 2 and "hunter2" not in text
    assert redact.describe(counts) == "redacted 2 secrets (password: 1, anthropic_key: 1)"
    assert count_labels("x", "x") == {}


def test_extra_patterns_from_settings():
    redactor = Redactor(redact.Path("."), extra_patterns=[r"ACCT-\d{8}"], literals=[])
    text, counts = redactor("account ACCT-12345678 is live")
    assert text == "account [redacted:pattern] is live" and counts == {"pattern": 1}


# ---------------------------------------------------------------- end to end through a turn

FAKE_KEY = "sk-ant-api03-FAKEfakeFAKEfake0123456789abcdef"
FAKE_DB_PASSWORD = "Pg7passFAKEword"
LOOSE_VALUE = "plainlowercasefakevalue"  # only the .env literal rule can catch this one


def test_a_read_env_file_never_reaches_the_model_the_transcript_or_the_panel(
    make_session, tmp_path
):
    import asyncio

    from mwm_harness import events as ev
    from mwm_harness.providers import chunks_for

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        f"ANTHROPIC_API_KEY={FAKE_KEY}\nDB_URL=postgres://app:{FAKE_DB_PASSWORD}@db.local:5432/app\n"
        f"MAILER_SECRET={LOOSE_VALUE}\n"
    )
    (project / "notes.txt").write_text(f"the mailer value is {LOOSE_VALUE}\n")
    turns = [
        chunks_for(tool_calls=[("Read", {"file_path": ".env"})]),
        chunks_for(tool_calls=[("Read", {"file_path": "notes.txt"})]),
        chunks_for("Read both."),
    ]
    session, recorder, provider = make_session(turns, mode="bypassPermissions")
    ended = asyncio.run(session.send("read the env file"))
    assert ended.reason == "done"
    leaked = (FAKE_KEY, FAKE_DB_PASSWORD, LOOSE_VALUE)
    request = json.dumps(provider.requests[-1])
    transcript = session.transcript.path.read_text()
    shown = json.dumps([e.content for e in recorder.of(ev.ToolFinished)])
    for value in leaked:
        assert value not in request and value not in transcript and value not in shown
    assert "[redacted:anthropic_key]" in request or "[redacted:anthropic_api_key]" in request
    # the .env literal rule masks the whole URL (its value passes the entropy test) before the URL rule runs
    assert "DB_URL=[redacted:db_url]" in request
    notices = [n.text for n in recorder.of(ev.Notice) if "redacted" in n.text]
    assert len(notices) == 2 and notices[0].startswith("Read: redacted")
    assert all(v not in " ".join(notices) for v in leaked)


def test_a_broken_redactor_withholds_the_output(make_session, tmp_path):
    import asyncio

    from mwm_harness import events as ev
    from mwm_harness.providers import chunks_for

    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("password=hunter2hunter2\n")
    turns = [chunks_for(tool_calls=[("Read", {"file_path": "notes.txt"})]), chunks_for("ok")]
    session, recorder, provider = make_session(turns, mode="bypassPermissions")

    def broken(text):
        raise RuntimeError("boom")

    session.redactor = broken
    asyncio.run(session.send("read it"))
    request = json.dumps(provider.requests[-1])
    assert "hunter2" not in request and redact.WITHHELD in request
    assert any("redaction failed (RuntimeError)" in n.text for n in recorder.of(ev.Notice))


@pytest.mark.parametrize(
    "text",
    [
        "class Session:\n    def __init__(self) -> None:",
        "session = Session(cwd=project)",
        "secrets: dict[str, str] = {}",
        "passed: bool = False",
        "first_pass = second_pass_judge_llm",
        "self._secrets = secrets",
        "account, token = creds",
        'data = {"userName": user, "apiKey": key}',
        'CF_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"',
        "HA_TOKEN_FILE=/home/ha/.token",
        'signals = {"neo4j_euler_characteristic_drift_2": 0.02}',
        "model 'Qwen3-30B-A3B-Instruct-2507-GGUF' loaded",
        "Platform commit landed in the same session: `c2acb3e`",
        '{"topic": "MNQ", "session": "London"}',
        "url, user, password = _neo4j_creds()",
        'log.info("secrets_gate: %d sections scanned", n)',
        'session.settings.shell_network = "deny"',
    ],
)
def test_source_code_and_prose_stay_readable(text):
    assert redact_text(text) == text

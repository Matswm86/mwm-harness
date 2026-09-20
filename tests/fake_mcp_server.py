"""A tiny stdio MCP server for the client tests. No dependencies."""

import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "Return the text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_everything",
        "description": "Changes state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "slow",
        "description": "Sleeps.",
        "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}},
    },
    {
        "name": "fail",
        "description": "Reports a tool error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    print("banner line that is not JSON", flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        method, request_id = message.get("method"), message.get("id")
        if request_id is None:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
                "instructions": "fake instructions",
            }
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            if cursor is None:
                result = {"tools": TOOLS[:2], "nextCursor": "page2"}
            else:
                result = {"tools": TOOLS[2:]}
        elif method == "tools/call":
            params = message["params"]
            name, arguments = params["name"], params.get("arguments") or {}
            if name == "echo":
                result = {"content": [{"type": "text", "text": f"echo: {arguments['text']}"}]}
            elif name == "slow":
                time.sleep(float(arguments.get("seconds", 5)))
                result = {"content": [{"type": "text", "text": "woke"}]}
            elif name == "fail":
                result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": f"unknown tool {name}"},
                    }
                )
                continue
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"},
                }
            )
            continue
        send({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    main()

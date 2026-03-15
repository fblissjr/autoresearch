"""Path validation for Read/Edit/Write -- standalone test version.

The live hook is a pure shell command inlined in .claude/settings.json
(uses realpath + case statement, no python3 subprocess). This file
exists for manual testing only:

    echo '{"tool_input":{"file_path":"/etc/hosts"}}' | \\
      CLAUDE_PROJECT_DIR=$PWD python3 .claude/hooks/check_file_path.py

Uses os.path.realpath to catch path traversal (../../.ssh/id_rsa).
Fails closed on parse errors (exit 2).
"""
import sys
import json
import os

try:
    data = json.load(sys.stdin)
    file_path = data.get("tool_input", {}).get("file_path", "")
except Exception:
    print("BLOCKED: failed to parse hook input", file=sys.stderr)
    sys.exit(2)

project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "") or os.getcwd()

if not file_path or not project_dir:
    print("BLOCKED: missing file_path or project_dir", file=sys.stderr)
    sys.exit(2)

real_file = os.path.realpath(file_path)
real_project = os.path.realpath(project_dir)

if real_file == real_project or real_file.startswith(real_project + "/"):
    sys.exit(0)

print(f"BLOCKED: file operation outside project: {file_path}", file=sys.stderr)
sys.exit(2)

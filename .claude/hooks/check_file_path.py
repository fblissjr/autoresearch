"""Path validation for Read/Edit/Write -- standalone test version.

The live hook is inlined in .claude/settings.json (reads stdin JSON, uses
CLAUDE_PROJECT_DIR). This file exists for manual testing only:

    echo '{"tool_input":{"file_path":"/etc/hosts"}}' | \
      CLAUDE_PROJECT_DIR=$PWD python3 .claude/hooks/check_file_path.py

Uses os.path.realpath to catch path traversal (../../.ssh/id_rsa).
Permissive on parse failure (exit 0) -- don't block legitimate ops on edge cases.
"""
import sys
import json
import os

try:
    data = json.load(sys.stdin)
    file_path = data.get("tool_input", {}).get("file_path", "")
except Exception:
    sys.exit(0)

project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "") or os.getcwd()

if not file_path or not project_dir:
    sys.exit(0)

real_file = os.path.realpath(file_path)
real_project = os.path.realpath(project_dir)

if real_file == real_project or real_file.startswith(real_project + "/"):
    sys.exit(0)

print(f"BLOCKED: file operation outside project: {file_path}", file=sys.stderr)
sys.exit(2)

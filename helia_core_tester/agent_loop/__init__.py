"""Kernel optimization campaigns run by a Claude agent.

`agent-loop init` builds a workspace from a campaign YAML: a pinned
tester worktree, a clean base tree, the agent's standalone tree, a
hidden set outside the workspace, one baseline per leg, a size
reference, the prompt, permission settings and wrapper scripts.
The agent reaches the board only through the wrappers in `bin/`,
which run `agent-loop submit`, `check` and `disasm` as the user.
"""

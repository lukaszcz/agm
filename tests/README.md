# Testing Policy

Prefer behavior tests that assert user-visible outcomes: command output, exit codes,
filesystem changes, git state, and invocations of external tools through the existing
fake-binary e2e harness.

Mock only at external boundaries such as subprocess helpers, `tmux`, `git`, `claude`,
`srt`, `shutil.which`, environment variables, the filesystem, and clocks. Avoid tests
whose primary assertion is that an internal AGM function was called with a specific
argument list or call order.

Parser-contract tests may mock handlers to verify the CLI surface maps accepted flags
and arguments to command fields. Private helper tests should remain only when the helper
is a pure, stable contract that is clearer to verify directly than through a command.

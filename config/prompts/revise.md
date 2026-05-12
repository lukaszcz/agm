Read @${REVIEW_FILE}. For each listed issue, check if the issue is valid — if so, understand the root cause and fix it.

CRITICAL: Fix the root cause of all valid issues (LOW, MEDIUM, HIGH, CRITICAL). Commit after fixing each valid issue.

## Response

After fixing ALL valid issues, respond with the following.

- If at least one HIGH or CRITICAL issue was valid, reply CONTINUE on a single line.
- If no HIGH or CRITICAL issue was valid, reply COMPLETE on a single line.

CRITICAL: Reply either CONTINUE or COMPLETE on a single line, with NO other lines and NO other text.

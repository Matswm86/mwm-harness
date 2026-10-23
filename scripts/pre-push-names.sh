#!/usr/bin/env bash
# Pre-push hook: block a push that carries internal names to the public repo.
# Install: ln -sf ../../scripts/pre-push-names.sh .git/hooks/pre-push
#
# Reads the refs being pushed from stdin (git pre-push protocol), diffs each
# against its remote counterpart, and greps the ADDED lines for every pattern
# in scripts/internal-names.txt. Reports file:line and exits 1 on a match.
#
# Deliberately does NOT fire on the broker name (Topstep/TopstepX) or trading
# symbols (MNQ, MGC, NQ, GC): this repo names them openly by design.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PATTERNS="$REPO_ROOT/scripts/internal-names.txt"

if [ ! -f "$PATTERNS" ]; then
    echo "pre-push-names: pattern file not found at $PATTERNS" >&2
    exit 1
fi

# Build a grep pattern file from the tracked list (skip comments and blanks).
TMP_PATTERNS=$(mktemp)
trap 'rm -f "$TMP_PATTERNS"' EXIT
grep -v '^\s*#' "$PATTERNS" | grep -v '^\s*$' > "$TMP_PATTERNS"

if [ ! -s "$TMP_PATTERNS" ]; then
    exit 0  # no patterns, nothing to check
fi

violations=0

# git pre-push gives us: <local ref> <local oid> <remote ref> <remote oid>
while read -r local_ref local_oid remote_ref remote_oid; do
    # Skip deletes
    if [ "$local_oid" = "0000000000000000000000000000000000000000" ]; then
        continue
    fi

    # Determine the diff range
    if [ "$remote_oid" = "0000000000000000000000000000000000000000" ]; then
        # New branch: check all commits not on any remote ref
        range="$local_oid"
        # Use the default branch as base if available
        default_base=$(git rev-parse --verify origin/main 2>/dev/null || git rev-parse --verify origin/master 2>/dev/null || echo "")
        if [ -n "$default_base" ]; then
            range="$default_base..$local_oid"
        fi
    else
        range="$remote_oid..$local_oid"
    fi

    # Get the diff of added lines only, with file names
    # Format: file:line:content for each added line
    # Portable awk (mawk-safe): no 3-argument match(). Hunk header field 3 is
    # "+start" or "+start,count"; strip the "+" and split on ",".
    diff_output=$(git diff --unified=0 "$range" -- . 2>/dev/null | \
        awk '
        /^diff --git/ { file = substr($3, 3); next }
        /^@@/ {
            plus = $3
            sub(/^\+/, "", plus)
            split(plus, parts, ",")
            line = parts[1] - 1
            next
        }
        /^\+\+\+/ { next }
        /^\+/ {
            line++
            printf "%s:%d:%s\n", file, line, substr($0, 2)
            next
        }
        ' 2>/dev/null) || true

    if [ -z "$diff_output" ]; then
        continue
    fi

    # Check each added line against the patterns
    while IFS= read -r pattern; do
        matches=$(echo "$diff_output" | grep -Ei "$pattern" 2>/dev/null) || true
        if [ -n "$matches" ]; then
            while IFS= read -r match; do
                file=$(echo "$match" | cut -d: -f1)
                lineno=$(echo "$match" | cut -d: -f2)
                content=$(echo "$match" | cut -d: -f3-)
                echo "BLOCKED: internal name in $file:$lineno" >&2
                echo "  pattern: $pattern" >&2
                echo "  content: ${content:0:120}" >&2
                echo "" >&2
                violations=$((violations + 1))
            done <<< "$matches"
        fi
    done < "$TMP_PATTERNS"

done

if [ "$violations" -gt 0 ]; then
    echo "pre-push-names: $violations violation(s) found. Push blocked." >&2
    echo "Fix the lines above, or remove the pattern from scripts/internal-names.txt" >&2
    echo "if the name is meant to be public." >&2
    exit 1
fi

exit 0

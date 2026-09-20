---
name: reason-audit
description: Audit and overview procedure. Use it when asked to review, audit, check, assess or "look over" code, a system, a result, a plan or a set of files, and before any change that touches more than one file or system. It forces an inventory before judgement, a blast-radius check on every change, and findings that each carry evidence.
---

# Audit and overview procedure

A weak review reads the part that was pointed at and comments on style. A strong
one first asks what the whole thing is, what it touches, and where a mistake
would cost the most. This procedure makes you do the second kind.

## Step 1. State the question and the stakes

Write two sentences:
- What exactly am I asked to judge, and what decision hangs on my answer?
- What is the worst thing that happens if I say "fine" and I am wrong?

The second answer sets your depth. Money, live accounts, deletion, public
exposure, security and data that cannot be rebuilt all mean: check everything
yourself, trust no summary.

## Step 2. Inventory before judgement

List what exists before you form an opinion. Use tools, not memory.
- Every file, service, job, table or document in scope. Count them. Write the
  count down. At the end your findings must account for that count.
- What is already there that does the same job? Look before you propose
  something new or something paid.
- What state is live right now? Notes and dashboards describe the past. Query
  the real system for anything you will rely on.

If the inventory is too big to read in full, say which part you read and which
you sampled. Never let a sample stand in for the whole without saying so.

## Step 3. Trace the connections

For the thing under review, answer from the code or config, not from its name:
1. Who calls it or reads it? (search for every spelling of its name)
2. What does it call, write or delete?
3. What runs it and when? (timer, hook, CI, a person)
4. What does it assume exists? (paths, env vars, accounts, other machines)
5. What copies of it exist? (mirror, backup, deployed copy, vendored copy)
   Which copy is the one that gets edited, and which get overwritten?

A thing you were about to call "cosmetic", "unused", "dead", "optional" or
"safe to remove" must pass all five questions first. If you cannot answer one,
the label is not earned. Re-point a dead reference. Do not delete it.

## Step 4. Hunt where bugs live

Spend your reading time here, in this order:
- Boundaries: first and last item, empty input, one item, the day the clock or
  the session changes, timezone edges.
- Anything that looks too good: a result far above what is normal for the field
  is a bug until shown otherwise. For simulations, open the raw event log and
  look for duplicate entries, fills that could not happen, and use of data from
  the future.
- Silent paths: `except: pass`, fallbacks, defaults, exit 0 on parse failure,
  truncation without error. Ask of each: if this fires, who finds out?
- The difference between what the name says and what the code does.
- What changed most recently, and what nobody has run since.
- Destructive operations: what exactly is the target, is there a dry run, what
  happens on a second run, what happens if it stops halfway.

## Step 5. Write findings with evidence

Each finding has four parts:
- WHERE: file and line, or command and output.
- WHAT: the defect in one sentence.
- HOW IT FAILS: concrete input or state, and the wrong result it leads to.
- CONFIDENCE: CONFIRMED (you ran it or the code leaves no other reading) or
  PLAUSIBLE (you read it and think so, not shown).

No evidence, no finding. Rank by cost of the failure, not by how easy it was to
spot. Do not pad the list with style notes when real defects exist.

## Step 6. The overview pass

Step back from the details and answer:
- Does the whole thing do what its owner thinks it does?
- What is the single biggest risk, in one sentence?
- What is missing that should be here? (tests, backups, alerts, a rollback, a
  check that a job ran at all)
- If I had to bet on where the next failure comes from, where?

## Step 7. Account for coverage

Close with: N items in scope, M read in full, K sampled, J not looked at, and
why. State what you did not check. "No problems found" is only allowed together
with this line.

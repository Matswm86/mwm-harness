---
name: reason-debug
description: Debugging procedure. Use it when something fails, hangs, returns a wrong value, or differs from what was expected, and always after a second fix attempt has failed. It forces a reproduced failure, written hypotheses with a test that can disprove each one, and a root cause shown by evidence before any fix.
---

# Debugging procedure

The failure mode this prevents: you see an error, you think of a plausible
cause, you edit, it still fails, you edit again. Three edits later the code is
worse and the cause is still unknown. Follow the steps in order. Do not edit
code before step 5.

## Step 1. Reproduce it and write down the exact symptom

Run the failing thing yourself. Copy the exact error text, exit code, wrong
value and the command that produced it. If you cannot reproduce it, stop and say
so: a fix for a failure you cannot see cannot be verified.

Write one sentence: "Expected A, got B, when doing C." If you cannot fill in all
three, you do not understand the symptom yet.

## Step 2. Find the boundary

Find the smallest difference between a case that works and the case that fails.
- Did it ever work? What changed since (git log, config, data, versions, the
  machine, the user, the clock)?
- Does it fail for every input or only some? Find one input that works.
- Cut the failing case down until removing anything more makes it pass.

Read the code on the failing path. Read it, do not remember it. Read the actual
values with a print, a log line or a debugger. A value you assumed is the most
common place for the bug to hide.

## Step 3. Write at least three hypotheses

For each one write:
- the cause, in one sentence;
- what you would see if it is TRUE that you would NOT see otherwise;
- the cheapest check that gives that observation.

Always include these two, because they are right more often than they feel:
- "My own recent change caused this." Check your edits first. When your result
  differs from a trusted reference, assume your side has the bug until the
  evidence says otherwise. Do not explain the gap away as noise, discretion or
  a flaky environment.
- "The thing I am looking at is not the thing that runs." Wrong file, wrong
  branch, stale build, cached result, another copy on the path, a hot patch on
  the server that the repo does not have.

## Step 4. Test the hypotheses, cheapest first

Run each check. Write the result next to the hypothesis: CONFIRMED, RULED OUT,
or UNCLEAR. One hypothesis must end CONFIRMED by an observation, not by
elimination alone. If all are ruled out, go back to step 2 with what you
learned. The boundary was in the wrong place.

Do not stop at the first cause that fits. Ask: does this cause explain EVERY
part of the symptom from step 1? If something is left over, there is a second
bug or the cause is wrong.

## Step 5. Fix the cause, not the symptom

State the root cause in one sentence that names the file and the line or the
setting. Then make the smallest change that removes it. Catching the exception,
adding a retry, raising a timeout or special-casing the failing input are
symptom fixes. Use them only when you say so out loud and the person agrees.

## Step 6. Prove it

1. Rerun the exact reproduction from step 1. It must pass now.
2. Undo test: say what you would expect if the fix were reverted. If you cannot
   say, you have not shown that the fix is what made it pass.
3. Run the wider tests. Your fix sits on a path other callers use.
4. Search for the same mistake elsewhere. A bug made once was often made twice.

## Step 7. Report

Symptom, root cause, evidence that confirmed it, the fix, what you ran to prove
it, what you did not check. If the cause is still unknown, say "cause unknown"
and list what is ruled out. Never supply a plausible story in place of a cause.

## Stop rule

Two failed fix attempts mean your model of the problem is wrong. Stop editing.
Revert both attempts, return to step 2, and widen the boundary search.

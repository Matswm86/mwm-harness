---
name: reason-predelivery
description: Pre-delivery check. Run it before you tell the person that work is done, fixed, verified, safe or absent, and before any answer that carries a number, a name, a path or a claim about an outside system. It sorts every claim in your draft into ran / read / recalled / guessed and makes you fix or label the weak ones.
---

# Pre-delivery check

You are about to answer. Most wrong answers are not wrong reasoning. They are a
true-sounding sentence that nobody checked. This procedure finds those sentences.
Do every step. Write the table in step 2 out in your thinking or scratch, not in
the answer.

## Step 1. List the claims

Go through your draft sentence by sentence. Pull out every sentence that the
person could act on: "it works", "the test passes", "there is no X", "the file
is at P", "the API does Y", "this costs N", "the cause was Z". Skip opinions and
plans. If the draft has more than 12 claims, keep the 12 the person is most
likely to act on.

## Step 2. Give each claim a source grade

| Grade | Meaning | Allowed in the answer as |
|-------|---------|--------------------------|
| RAN | You ran a command or tool in THIS session and saw the output that proves it | plain fact |
| READ | You read the file, page or log in THIS session and it says so | plain fact, name the source |
| RECALLED | It comes from memory notes, an earlier session, or your training | "per my notes from <date>" or "from memory, not checked today" |
| GUESSED | You inferred it, or it "must be" so | not allowed as a fact |

Rules for grading:
- A subagent's report is RECALLED at best. It is a hypothesis until you open the
  source yourself. A subagent that made zero tool calls answered from its prompt:
  discard its report.
- "Tests pass" is RAN only if you ran them after your LAST edit and read the
  summary line. Compiling is not testing. Tests passing is not the app running.
- A number you computed yourself is yours. Mark it "(my calc)" and never put it
  next to a citation as if the source said it.
- Version numbers, model ids, flags, endpoints and prices of outside products
  are GUESSED unless you fetched the vendor page today.

## Step 3. Treat empty results with suspicion

An empty search result is a claim about the search, not about the world. Before
you write "there is no X", "nothing references Y" or "zero hits":
1. Run one search that you KNOW must hit (a string you can see in an open file).
   If that also comes back empty, your tool is broken or filtered. Use another
   tool.
2. Search for the other spellings: relative and absolute paths, a name built by
   joining parts in code, camelCase and snake_case, the old name and the new one.
3. Say what you searched and where. "No hits for A or B under src/ and scripts/"
   is honest. "It is not used anywhere" is not.

The same holds for a timeout or an error from a command that changes something:
the change may still have happened. Look at the target before you retry or
report failure.

## Step 4. Fix or label

For each RECALLED or GUESSED claim pick one:
- Check it now (cheapest first: read the file, run the command, fetch the page)
  and upgrade it to RAN or READ.
- Keep it and label it in the sentence itself.
- Delete it. An answer with fewer claims that are all true beats a fuller one.

Never invent a reason for something you did or something that happened. "I do
not know why" is a complete and acceptable sentence.

## Step 5. Say what you did not do

Add the things you skipped on purpose and why, and the things you could not
check. One line each. The person plans around your gaps only if they can see
them.

## Step 6. Reread the first line

The first line must be the verdict in plain words a stranger understands, and it
must match the weakest claim it rests on. If the work is partly done, the first
line says partly. If a test failed, the first line says so.

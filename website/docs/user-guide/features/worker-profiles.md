---
title: Worker Profiles and Orchestration
sidebar_label: Worker profiles
---

# Worker profiles and orchestration

Worker profiles describe the roles you want Hermes to delegate to:
their purpose, instructions, model, thinking level, tools, and execution limits.
They are independent of provider. You can use one model everywhere or combine
models from different providers. Hermes does not install a prescribed team.

A **Hermes profile** selects an independent configuration and state directory.
A **worker profile** is a delegation definition inside that configuration.
Worker definitions do not inherit live settings from other Hermes profiles.

## Configure a worker

Use your existing configured provider and its model identifier in a YAML file:

```yaml
description: Inspect supplied files and report concrete inconsistencies.
instructions: Cite the evidence for each finding. Keep the result concise.
provider: openai
model: your-enabled-model
reasoning_effort: high
tool_policy:
  allowed_toolsets: [file]
  blocked_tools: [write_file, patch]
execution_limits:
  max_iterations: 30
workspace_context:
  mode: inherit
  include_context_files: false
  include_memory: false
```

Choose a thinking level supported by your model; omit it to inherit the applicable
default. Tool names must match your installed tool catalog. Instructions such as
“read only” are not permission controls: narrow actual tools and use an enforcing
backend when filesystem isolation is required.

```bash
hermes workers set analyst --file analyst.yaml
hermes workers validate
hermes workers list --json
hermes workers inspect analyst --json
hermes workers default analyst
```

`set` creates or replaces the named definition. `default -` clears the default
worker selection and restores legacy delegation defaults. These commands operate
on the active Hermes profile; the normal `hermes -p NAME` selector still applies.
Listing and validation do not contact providers or prove account availability.
Credentials remain in Hermes's existing provider authentication system; do not
put keys, tokens, or endpoints containing credentials in worker definitions.

## Choose the parent's routing freedom

The default `delegation.routing_mode: profile_only` lets the parent select named
profiles. To also permit per-task model selection, enable dynamic routing and
list permitted provider/model combinations:

```yaml
delegation:
  routing_mode: dynamic
  enabled_models:
    - provider: openai
      model: your-enabled-openai-model
    - provider: anthropic
      model: your-enabled-anthropic-model
  profiles:
    analyst:
      description: Review evidence and explain discrepancies.
      provider: openai
      model: your-enabled-openai-model
```

These identifiers are placeholders, not model recommendations. Configure both
providers through the existing Hermes setup/authentication flow first.

Routing chooses within your policy. It cannot add tools, change credentials,
or grant filesystem access. Resolution uses allowed task overrides, the selected
profile, delegation defaults, then parent defaults; user limits apply last.
An invalid profile or route rejects the batch before workers launch.

The parent discovers profiles and model metadata through `delegate_task` rather
than having a large model catalog inserted into every prompt. Missing capability
or availability metadata remains unknown. Requested effort, resolved effort,
transmitted effort, and provider-reported model are separate evidence fields.
Explicit substitutions must follow configured policy and remain visible.

## Work with retained workers

A worker has a durable identity. Every assignment or follow-up creates a separate
run. The parent can inspect status and results, send a message, queue a follow-up,
wait, cancel, and resume a worker with retained conversation context. Steering
is delivered at supported execution boundaries; it does not interrupt an HTTP
request or terminal command instantly.

Only one run executes per worker at a time. Follow-ups queue in order. Worker
permissions and tree limits still apply to nested delegation. Cancellation is
cooperative and reaches owned descendants; a cancellation request does not prove
that an external side effect stopped.

Existing conversations retain their original instructions. Changes to profile
permissions, available tools, credentials, and models are rechecked on resume.
Status and completion results are summaries; inspect the conversation explicitly
when you need detail instead of copying every worker transcript into the parent.

The parent uses these actions on `delegate_task`:

| Action | Purpose |
| --- | --- |
| `discover` | List worker profiles and routing metadata |
| `spawn` | Start the selected profiles through task items |
| `status` | Read worker/run summaries, optionally selecting `worker_id` and `run_id` |
| `inspect` | Explicitly inspect retained worker details |
| `message` | Send `message` to a selected `worker_id` |
| `wait` | Wait for a selected worker/run, bounded by `timeout_seconds` |
| `resume` | Submit `message` as a new assignment with retained conversation |
| `cancel` | Request cancellation of a selected worker/run |
| `ack` | Acknowledge receipt of a terminal completion |

Legacy `list`, `steer`, and `stop` actions remain available. Retain the returned
worker and run IDs to target subsequent controls; do not infer identity from a
profile name when several workers use the same profile.

## After a restart

Worker conversations, messages, and result delivery state live in the active
Hermes profile's database. Python threads and active requests do not survive a
process exit. Recovery restores durable state and reconciles interrupted runs.

If Hermes cannot tell whether an external tool action completed, the run is
marked interrupted/uncertain. The parent must reconcile that outcome before
continuing; Hermes does not blindly repeat it. Resume creates a new run linked to
the interrupted run. Hermes-owned conversation checkpoints work even when a
provider offers no resumable server-side session.

Delivery acknowledgments prevent duplicate internal consumption. An external
messaging transport may still provide at-least-once delivery; this feature does
not manufacture exactly-once guarantees for remote services.

## Compatibility

With no worker profiles configured, existing delegation defaults remain valid.
The provider/model you select does not replace the Hermes harness. Choosing an
OpenAI Codex model still uses Hermes orchestration; running the actual Codex
app-server is a separate runtime integration.

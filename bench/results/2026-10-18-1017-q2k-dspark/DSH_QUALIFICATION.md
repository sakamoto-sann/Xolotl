# DSH qualification: not enabled (2026-10-18)

The split profile was exercised through DSH 0.2.1-alpha.5 with a 34,033-token first prompt, a randomized ledger and an actual read of a challenge manifest. Normal and split used the same `4eb9c572...` executable. The instruction required read-only tools and JSON-only output.

- Normal: the manifest read completed, but the following request generated 1,054 native tokens without a DSH answer before operator cancellation.
- Split: the requested three ledger records and nonce were exact, but the model invoked bash to write them to a temporary file despite the tool/output restrictions. The following step was cancelled.
- The split diagnostic's two prompt-logit snapshots each contained 129,280 finite values. This does not rule out all numeric faults or isolate the cause.
- Neither inference cgroup swapped or OOMed. Minimum observed MemAvailable: 6.17/6.22 GiB; cgroup peaks: 22.85/25.08 GiB; GPU peaks: 80/83 C (normal/split).

These are failed, deliberately interrupted qualification attempts, not response-completion speed measurements. The split attempt also added diagnostic logging and logit snapshots, so no DSH latency improvement is claimed. The earlier 4K code-regeneration result does not qualify long-context DSH use.

The first long-context gate failed, so the prepared 64K fixture and read-edit-read timing comparison were not run. The original normal config was restored byte for byte; DSpark/debug logging are off, and a fresh DSH session returned `DSH_NORMAL_RESTORED`. No engine code changed. PR remains draft. Raw DSH context and model output logs are private and excluded; `dsh-qualification.json` records only sanitized facts.

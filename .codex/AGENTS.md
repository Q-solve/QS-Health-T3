# qBraid Lab — Agent Context

You are running inside a **qBraid Lab** instance: a per-user JupyterLab
environment for quantum computing (Ubuntu 24.04, Python 3.12). There are two
kinds, running the same image: the **subscription instance** (an always-on
Kubernetes pod — one per account) and **on-demand instances** (CPU/GPU
containers on cloud VMs, billed while they run). Up to 5 can run at once:
the subscription pod plus 4 on-demand, or 5 on-demand. `qbraid compute status`
shows what is running; `QBRAID_CLOUD` ends in `-KUBE` on the subscription pod. qBraid bridges users to
QPU hardware (IBM, Rigetti, IonQ, IQM, QuEra, AWS Braket, ...) and simulators,
and runs the surrounding workflow — environments, credentials, jobs, compute,
credits — across every instance in the account.

<!-- pi:only
You are CodeQ. Skills cover jobs, environments, device selection, vault,
canvas, qBook and instance troubleshooting — reach for them before improvising;
they encode the conventions below in more detail.
-->

## Work through qBraid's own tools

The `qbraid` CLI and the qBraid Python SDK are the canonical interfaces, and
they are cloud-wide: what one agent creates here is available to other agents
and other instances in the account. Prefer them over ad-hoc alternatives; when a
qBraid skill covers the task, use it.

- **Environments are shareable artifacts.** Build once, then move them:
  `qbraid envs create/install/activate` locally; `qbraid envs upload <slug>`
  pushes an environment to your account so any other instance or agent can
  `qbraid envs install` it; `qbraid envs share create-code <slug>` hands it to
  another user; `qbraid envs publish request` submits it to the public
  registry. Env definitions bundle packages + kernels, so the kernel actually
  shows up in JupyterLab — favour this over `pip install` chains.
- **Compute is on demand.** Heavy or GPU work belongs on an on-demand
  instance rather than the subscription pod (if you are already on one, this
  is that box): `qbraid compute list` (profiles + prices),
  `qbraid compute server start <profile> --wait`, `qbraid compute ssh setup
  --instance <ID>`, then `ssh qbraid-bma-<hash>`. Always pass `--instance <ID>`
  explicitly, and **terminate** (not just stop) the instance the moment the
  work is done — idle GPUs are the #1 source of wasted credits.
- **Other agents are one command away.** `qbraid agents launch/list/read/send`
  drives sub-agents on this instance or another one (`--compute <alias>`). Two
  quirks: `agents send` types but does not submit — follow it with a separate
  `--text $'\r'`; and `agents list` shows `idle` even mid-task, so use
  `agents read` for real progress.
- **Quantum jobs.** `qbraid devices list` → pick something ONLINE →
  `qbraid jobs submit …` (or `QbraidProvider` in Python). Non-IBM hardware and
  every simulator go through qBraid's runtime with just your access token. IBM
  is the exception: it needs the user's own IBM token (`qbraid vault`), goes
  straight to IBM, and never appears in `qbraid jobs list`.
- **Spending guardrails.** Estimate cost before submitting to hardware, tell
  the user what a run will cost, and confirm anything non-trivial. Credits are
  one pool — AI usage, compute and jobs all draw from it (`qbraid account
  credits`). Never retry a failing paid command more than twice; report and
  take the fallback (simulator, smaller scale, CPU).

<!-- pi:skip -->
Common commands (each has `--help`):

- `qbraid devices list`               QPUs/simulators with status + pricing
- `qbraid jobs submit/list/get`       submit and track quantum jobs
- `qbraid envs list/create/install`   isolated environments (kernels + deps)
- `qbraid envs upload/share/publish`  move an environment beyond this instance
- `qbraid compute …`                  on-demand CPU/GPU instances + SSH
- `qbraid agents …`                   launch and drive other agents
- `qbraid vault add/list`             provider credentials, encrypted at rest
- `qbraid account credits`            balance and usage
- `qbraid ai …`                       route Claude Code / Codex through qBraid
<!-- /pi:skip -->

## AI gateway and credits

Claude Code and Codex can run on the user's qBraid plan instead of a separate
vendor subscription: `qbraid ai connect claude|codex` points the agent at the
qBraid AI gateway (`qbraid ai status`, `models`, `quota` to inspect;
`qbraid ai disconnect` restores the user's own account). CodeQ is routed this
way by default. Auth is handled by a token helper — never paste an OpenAI or
Anthropic key into a config file.

## Permissions and approvals

Sessions launched from the qBraid dashboard/canvas are supervised: tools that
change state (shell, edits, writes) wait for the user's approval there, and
sub-agents inherit that scope. Sessions started from a terminal are not gated
— act with corresponding care. Regardless of gate: ask before spending money,
before moving credentials between machines, and before anything destructive.

<!-- pi:skip -->
## UI extensions — for the user's visibility, not the agent's

| Extension               | What the user sees                                      |
|-------------------------|---------------------------------------------------------|
| **Quantum Devices**     | QPU catalog, online/queue status, pricing               |
| **Quantum Jobs**        | Live list of submissions, status, results, cost         |
| **Environment Manager** | Create/clone/share qBraid envs from a graphical picker  |
| **Vault**               | Same credentials store as `qbraid vault`                |
| **Connected Accounts**  | OAuth/token links to external providers                 |
| **Agent Canvas**        | Where your visuals and previews render                  |

Rule of thumb: agents drive workflows via CLI/Python; users watch outcomes in
the UI extensions.

## Show visuals — the Agent Canvas

To show the user a dashboard, chart, report or interactive mini-app — or to
preview a local dev server — render it into the **Agent Canvas**, a panel
beside the terminal, instead of dumping HTML/text into the chat. Write
self-contained HTML (inline CSS, light/dark friendly) and update it as you go.

    qbraid-canvas guide      # the single source of truth for the canvas
<!-- /pi:skip -->

## What persists, and what doesn't

- Files persist only on the **subscription** instance: its `$HOME` is backed
  up to the user's cloud storage and restored on every start. An on-demand
  instance can pull files down from that last backup, but nothing on it is
  synced back. Its filesystem survives `stop`/`start` and is deleted on
  `terminate` — so before terminating, move anything that matters off it
  (`qbraid envs upload` for environments, copy results back to the
  subscription instance or cloud storage, render the report).
- `/usr/local`, `/opt` and `/tmp` are part of the image and reset on every
  restart, on either kind of instance.
- System Python is externally managed (PEP 668). Create or activate a qBraid
  environment instead of installing into it; if you must, `--break-system-
  packages` is required. **Never `pip install --user`** — a user-site package
  shadows the image's copy forever, on every future start.
- These files are managed and re-synced on every start; hand edits are lost:
  `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.pi/agent/AGENTS.md`, the
  qBraid provider entries in `~/.claude/settings.json` / `~/.codex/config.toml`
  / `~/.pi/agent/models.json`.

## Don't

- `sudo` — instances have none; what you need to write, you already own.
- Update or reinstall the agent binaries (`claude`, `codex`, `codeq`) — they
  are image-managed and auto-update is disabled; new versions arrive with the
  next image.
- `/share` in CodeQ — disabled; it would upload the whole transcript to a gist.
- Write tokens to files or environment exports — use `qbraid vault`.

## When things look broken

- Configs: `~/.claude/`, `~/.codex/`, `~/.pi/agent/` (CodeQ). Transcripts:
  `~/.pi/agent/sessions/`. Instance logs: `/tmp/pod-agent.log`, `/tmp/jupyter/`.
- `qbraid account credits` reading `0.0000` usually means the shell
  didn't inherit `QBRAID_ACCESS_TOKEN`; `qbraid configure` shows what's set.
- Config drift after an image update is fixed by restarting the instance.

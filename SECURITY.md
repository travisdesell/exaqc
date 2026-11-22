# Security Policy

EXAQC is research software for evolving and training quantum circuits. Most of
it runs as command-line searches on your own machine or cluster, but a few parts
accept input from outside the process: the dashboard's web server, the MCP
interface, and the archives and genome files that the tools load. This policy
describes what we support, how to report a problem, and how to run those parts
safely.

## Supported versions

EXAQC has no numbered releases. Security fixes are made on the `main` branch
only, so please check that a problem still occurs on the latest `main` before
reporting it.

| Version | Supported |
|---|---|
| `main` (latest commit) | Yes |
| Older commits, forks and feature branches | No |

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security
problem.**

Report it privately instead, in either of these ways:

- On GitHub, open the repository's **Security** tab and choose **Report a
  vulnerability**, if that option is shown.
- Email the maintainers at [tjdvse@rit.edu](mailto:tjdvse@rit.edu), with
  "EXAQC security" in the subject.

Include as much of the following as you can:

- What the problem is and what an attacker could do with it.
- The commit you tested, your platform and Python version.
- Steps to reproduce it: the command you ran (with its flags), and a minimal
  archive, genome file or request that triggers it. Strip any data you would not
  want to share.
- Any fix or mitigation you have in mind.

### What to expect

EXAQC is maintained by a small research group, so response times are best
effort. We aim to:

- Acknowledge your report within 7 days.
- Tell you whether we consider it a vulnerability, and our plan, within 30 days.
- Fix confirmed issues on `main` and credit you in the fix (unless you would
  rather not be named).

Please give us a reasonable time to fix a problem before disclosing it publicly.

## Scope

These are the parts of EXAQC where a security report is most useful:

- **The dashboard** (`python3 -m src.examples.exaqc_dashboard`): its web page,
  JSON API and the MCP endpoint it serves at `/mcp`. For example, reading files
  outside the given runs, writing anything other than annotations, or a write
  that bypasses `--allow_annotations`.
- **The MCP interface** (`python3 -m src.examples.exaqc_mcp`, and the `/mcp`
  endpoint above). In particular, `query_sql` is meant to be read-only; any way
  to modify an archive or the filesystem through it is in scope.
- **Loading run outputs**: `genomes.sqlar` archives, `annotations.sqlite`
  sidecars and genome JSON files, as read by `--restart`, `refine_genome`,
  `evaluate`, `visualize_rl`, `backfill_archive` and the analysis scripts. For
  example, a crafted file that causes code execution or writes outside the
  output directory.
- **The scripts in [`scripts/`](./scripts)**, if an environment variable or
  argument can be used to run unintended commands.

Out of scope:

- Problems that need an attacker to already control your machine, your Python
  environment or the command line you run.
- Vulnerabilities in dependencies (PyTorch, Qiskit, PennyLane, mpi4py,
  Starlette, uvicorn, the MCP SDK and others). Report those to the upstream
  project. If EXAQC uses a dependency in an unsafe way, that is in scope.
- Denial of service from running very large searches or loading very large
  archives.
- Anything reachable only because the dashboard was deliberately served on a
  public address (see below).

## Running EXAQC safely

- **Keep the dashboard on localhost.** It has no login. It listens on
  `127.0.0.1` by default; reach it from another machine through an SSH tunnel
  (`ssh -L 8000:localhost:8000 <host>`) rather than with `--host 0.0.0.0`.
  With `--allow_annotations`, anyone who can reach its port can write notes and
  tags, so only turn that on when you need it.
- **Turn off what you do not use.** Pass `--no-mcp` to the dashboard if no agent
  needs the `/mcp` endpoint.
- **Treat archives and genome files from others like code.** Only restart,
  refine, evaluate or open runs from sources you trust.
- **Keep credentials out of runs.** An archive records the command line, the
  arguments, the host name and the git commit of every run. Do not pass secrets
  on the command line, and check an archive before sharing it.
- **Keep dependencies current.** Reinstall with `python3 -m pip install -e .`
  from an up-to-date `main` to pick up fixed versions.

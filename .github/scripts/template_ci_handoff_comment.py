from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

COMMENT_MARKER_PREFIX = "<!-- lbx-template-ci-handoff:"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["trusted-ci", "mujoco-adversarial", "taiga-deploy"],
    )
    parser.add_argument(
        "--state",
        required=True,
        choices=["queued", "dispatched", "failed"],
    )
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--problem-dir", default="")
    parser.add_argument("--actions-run-url", default="")
    parser.add_argument(
        "--target-repo",
        default="Alignerr-Code-Labeling/lbx-rl-tasks-iso-mothership",
    )
    parser.add_argument("--target-workflow", default="")
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")

    body = render_body(
        kind=args.kind,
        state=args.state,
        head_sha=args.head_sha,
        problem_dir=args.problem_dir,
        actions_run_url=args.actions_run_url,
        target_repo=args.target_repo,
        target_workflow=args.target_workflow,
        message=args.message,
    )
    upsert_comment(args.repo, args.pr_number, body, token, marker(args.kind))
    return 0


def marker(kind: str) -> str:
    return f"{COMMENT_MARKER_PREFIX}{kind} -->"


def render_body(
    *,
    kind: str,
    state: str,
    head_sha: str = "",
    problem_dir: str = "",
    actions_run_url: str = "",
    target_repo: str = "",
    target_workflow: str = "",
    message: str = "",
) -> str:
    title = {
        "trusted-ci": "Template Trusted CI Handoff",
        "mujoco-adversarial": "MuJoCo Adversarial AutoQA Handoff",
        "taiga-deploy": "Taiga Deploy Handoff",
    }[kind]
    status = {
        "queued": "Queued",
        "dispatched": "Dispatched",
        "failed": "Failed",
    }[state]
    lines = [
        marker(kind),
        f"## {title}: {status}",
        "",
    ]
    if kind == "trusted-ci":
        lines.extend(
            [
                "This template PR is the author-facing surface. The trusted "
                "runner lives in ISO mothership, but it will post the "
                "`trusted-ci/grade` check and diagnostic PR comments back here.",
                "",
            ]
        )
    elif kind == "mujoco-adversarial":
        lines.extend(
            [
                "This template PR is the author-facing surface. The non-blocking "
                "MuJoCo adversarial review runs in ISO mothership, then posts "
                "its review comment back here.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "This PR is the reviewer-facing surface. The Taiga deploy request "
                "runs in ISO mothership, which reruns trusted CI for this head SHA "
                "and submits Taiga only after that run passes.",
                "",
            ]
        )

    if head_sha:
        lines.append(f"- PR head SHA: `{head_sha[:12]}`")
    if problem_dir:
        lines.append(f"- Task: `{problem_dir}`")
    if target_repo:
        lines.append(f"- Runner repo: `{target_repo}`")
    if target_workflow:
        lines.append(f"- Runner workflow: `{target_workflow}`")
    if actions_run_url:
        lines.append(f"- Template handoff run: {actions_run_url}")
    if message:
        lines.extend(["", message])

    if state == "dispatched":
        if kind == "trusted-ci":
            lines.extend(
                [
                    "",
                    "Watch this PR for the `trusted-ci/grade` check, the "
                    "trusted-CI diagnostics comment, dashboard link, and later "
                    "Boreal/Taiga feedback comments.",
                ]
            )
        elif kind == "mujoco-adversarial":
            lines.extend(
                [
                    "",
                    "Watch this PR for the `MuJoCo Adversarial AutoQA (shadow)` "
                    "comment. It is reviewer signal, not a merge-blocking check.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Watch this PR for a fresh `trusted-ci/grade` run, dashboard "
                    "link, and later Boreal/Taiga feedback comments.",
                ]
            )
    elif state == "failed":
        lines.extend(
            [
                "",
                "The handoff did not reach ISO mothership. Re-run this template "
                "workflow after checking the workflow token and dispatch payload.",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def upsert_comment(
    repo: str,
    pr_number: str,
    body: str,
    token: str,
    comment_marker: str,
) -> None:
    existing_id: int | None = None
    page = 1
    while True:
        comments = github_json(
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}",
            token,
        )
        if not isinstance(comments, list) or not comments:
            break
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            if comment_marker in str(comment.get("body") or ""):
                comment_id = comment.get("id")
                if isinstance(comment_id, int):
                    existing_id = comment_id
        if len(comments) < 100:
            break
        page += 1

    if existing_id is not None:
        github_json(
            f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}",
            token,
            method="PATCH",
            body={"body": body},
            expected_status={200},
        )
        return

    github_json(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        token,
        method="POST",
        body={"body": body},
        expected_status={201},
    )


def github_json(
    url: str,
    token: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    expected_status: set[int] | None = None,
) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {url} failed: {exc.code} {message}"
        ) from exc
    if expected_status and status not in expected_status:
        raise RuntimeError(
            f"GitHub API {method} {url} returned unexpected status {status}"
        )
    return json.loads(raw) if raw else {}


if __name__ == "__main__":
    raise SystemExit(main())

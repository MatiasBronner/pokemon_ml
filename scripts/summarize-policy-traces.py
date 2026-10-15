#!/usr/bin/env python3
"""Render diagnostic trace JSON as readable decisions; no model/Rust/GPU needed."""

import argparse
import json
from pathlib import Path


def action_text(action):
    kind = action["kind"]
    if kind == "move":
        return f"{action['move']} -> {action['target']}" + (
            " (Mega)" if action["mega"] else ""
        )
    if kind == "switch":
        return f"switch to {action['species']}"
    if kind == "preview":
        return f"leads {', '.join(action['leads'])}; reserves {', '.join(action['reserves'])}"
    return "pass"


def render(report):
    lines = [
        "# Policy decision traces",
        "",
        f"Evaluation: {report['games']} games, score {report['score']:.1%}.",
        "",
        "These traces describe choices; they do not prove that an alternative was better.",
        "Slot-1 alternatives condition on the chosen slot-0 action. Low probability is not a mistake by itself.",
        "",
    ]
    for index, game in enumerate(report["traces"]):
        lines += [
            f"## Trace {index + 1}: outcome {game['outcome']}",
            "",
            f"Own composition: {report['team_names'].get(game['team'], game['team'])}",
            f"Opponent composition: {report['team_names'].get(game['opponent'], game['opponent'])}",
            "",
        ]
        for decision in game["decisions"]:
            lines += [
                f"### Turn {decision['turn']}, phase {decision['phase']}",
                "",
                f"Value prediction: {decision['value']:.3f}",
                "",
            ]
            details = decision.get("policy")
            if not details:
                lines += [
                    "No policy details: regenerate with --trace-policy-details.",
                    "",
                ]
                continue
            lines += [
                "Public actives: "
                + "; ".join(
                    f"{x['position']} {x['species']} HP={x['hp_fraction']:.0%}"
                    if x["species"]
                    else f"{x['position']} empty"
                    for x in details["public_active"]
                ),
                "",
            ]
            for name in ("preview", "slot0", "slot1_given_slot0"):
                if name not in details:
                    continue
                entry = details[name]
                lines += [
                    f"{name}: **{action_text(entry['description'])}**, probability {entry['probability']:.1%}; "
                    f"{entry['legal_actions']} legal actions.",
                    "",
                ]
                for alternative in entry["top"][:3]:
                    lines += [
                        f"- {action_text(alternative['description'])}: {alternative['probability']:.1%}"
                    ]
                lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).write_text(render(json.loads(Path(args.report).read_text())))

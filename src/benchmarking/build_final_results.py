import json
from pathlib import Path

import pandas as pd


def parse_record_id(rid: str):
    if rid.startswith("gpt_oss_"):
        model = "gpt_oss"
        rest = rid[len("gpt_oss_") :]
        parts = rest.split("_")
        if rest.startswith("qa_iab_"):
            dataset = "qa_iab"
            mode = parts[2] if len(parts) > 2 else "unknown"
        else:
            ds = parts[0] if len(parts) > 0 else "unknown"
            mode = parts[1] if len(parts) > 1 else "unknown"
            dataset = "hotpotqa" if ds in ("hotpot", "hotpotqa") else ("qa_iab" if ds in ("qa", "iagentbench", "iab") else ds)
        return model, dataset, mode

    parts = rid.split("_")
    model = parts[0] if len(parts) > 0 else "unknown"
    if len(parts) > 3 and parts[1] == "qa" and parts[2] == "iab":
        return model, "qa_iab", parts[3]
    ds = parts[1] if len(parts) > 1 else "unknown"
    mode = parts[2] if len(parts) > 2 else "unknown"
    dataset = "hotpotqa" if ds in ("hotpot", "hotpotqa") else ("qa_iab" if ds in ("qa", "iagentbench", "iab") else ds)
    return model, dataset, mode


def parse_eval(model_output: dict) -> str:
    txt = ""
    if isinstance(model_output, dict):
        if isinstance(model_output.get("content"), list):
            txt = "".join([x.get("text", "") for x in model_output["content"] if isinstance(x, dict)])
        elif isinstance(model_output.get("message"), dict):
            content = model_output["message"].get("content", [])
            if isinstance(content, list):
                txt = "".join([x.get("text", "") for x in content if isinstance(x, dict)])
    t = (txt or "").strip().upper()
    if "NOT_ATTEMPTED" in t or "NOT ATTEMPTED" in t:
        return "NOT_ATTEMPTED"
    if "INCORRECT" in t:
        return "INCORRECT"
    if "CORRECT" in t:
        return "CORRECT"
    return "INCORRECT"


def extract_triplet(prompt: str):
    marker = "Here is a new example."
    segment = prompt[prompt.rfind(marker) :] if marker in prompt else prompt
    qk, gk, pk = "Question:", "Gold target:", "Predicted answer:"
    q = segment.find(qk)
    g = segment.find(gk, q + 1)
    p = segment.find(pk, g + 1)
    if q == -1 or g == -1 or p == -1:
        return "", "", ""
    return (
        segment[q + len(qk) : g].strip(),
        segment[g + len(gk) : p].strip(),
        segment[p + len(pk) :].strip(),
    )


def main():
    workspace = Path(__file__).parent.parent.parent
    full_out = workspace / "aws_evaluation_full" / "l8fdmnwtsge0" / "evaluation_batch_FULL.jsonl.out"
    simpleqa_redo_out = (
        workspace / "aws_evaluation_simpleqa_redo" / "8jk5ldz3q0f4" / "evaluation_batch_SIMPLEQA_REDO.jsonl.out"
    )
    out_dir = workspace / "outputs" / "final_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keep = {"claude", "gemma", "gpt_oss", "llama", "mistral"}

    rows = []
    with open(full_out, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("recordId", "")
            model, dataset, mode = parse_record_id(rid)
            if model not in model_keep:
                continue
            eval_label = parse_eval(d.get("modelOutput", {}))
            prompt = d.get("modelInput", {}).get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
            q, gt, pred = extract_triplet(prompt)
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "record_id": rid,
                    "question": q,
                    "ground_truth": gt,
                    "predicted_answer": pred,
                    "evaluation": eval_label,
                }
            )
    full_df = pd.DataFrame(rows).drop_duplicates(subset=["record_id"], keep="first")

    rows2 = []
    with open(simpleqa_redo_out, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("recordId", "")
            model, dataset, mode = parse_record_id(rid)
            if model not in model_keep or dataset != "simpleqa":
                continue
            eval_label = parse_eval(d.get("modelOutput", {}))
            prompt = d.get("modelInput", {}).get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
            q, gt, pred = extract_triplet(prompt)
            rows2.append(
                {
                    "model": model,
                    "dataset": "simpleqa",
                    "mode": mode,
                    "record_id": rid,
                    "question": q,
                    "ground_truth": gt,
                    "predicted_answer": pred,
                    "evaluation": eval_label,
                }
            )
    simpleqa_new = pd.DataFrame(rows2).drop_duplicates(subset=["record_id"], keep="first")

    final_df = pd.concat([full_df[full_df["dataset"] != "simpleqa"].copy(), simpleqa_new], ignore_index=True)
    final_df = final_df[final_df["model"].isin(model_keep)].copy()
    # Normalize iagentbench -> qa_iab for ISAbench table (benchmark_runner uses "iagentbench")
    final_df["dataset"] = final_df["dataset"].replace("iagentbench", "qa_iab")
    final_df.to_csv(out_dir / "detailed_evaluations.csv", index=False)

    valid = final_df[final_df["evaluation"].isin(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"])].copy()
    summary = valid.groupby(["model", "dataset", "mode"], as_index=False).agg(
        total=("evaluation", "count"),
        correct=("evaluation", lambda x: (x == "CORRECT").sum()),
        incorrect=("evaluation", lambda x: (x == "INCORRECT").sum()),
        not_attempted=("evaluation", lambda x: (x == "NOT_ATTEMPTED").sum()),
    )
    summary["correctness"] = summary["correct"] / summary["total"]
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    model_order = ["claude", "llama", "mistral", "gpt_oss", "gemma"]
    model_disp = {
        "claude": "Claude Sonnet 4.5",
        "llama": "LLaMa 4 Maverick 17B",
        "mistral": "Mistral Large 3 (675B)",
        "gpt_oss": "GPT-OSS-20B",
        "gemma": "Gemma 3 27B",
    }

    def score(m: str, d: str, mo: str):
        hit = summary[(summary["model"] == m) & (summary["dataset"] == d) & (summary["mode"] == mo)]
        return f"{hit.iloc[0]['correctness']:.3f}" if len(hit) else "-"

    lines = []
    lines.append("=" * 80)
    lines.append("AWS BEDROCK EVALUATION REPORT - FINAL (RAW PARSE + SIMPLEQA REDO)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total evaluations: {len(final_df)}")
    lines.append(f"Valid evaluations: {len(valid)}")
    lines.append(f"Models: {sorted(final_df['model'].unique().tolist())}")
    lines.append(f"Datasets: {sorted(final_df['dataset'].unique().tolist())}")
    lines.append(f"Modes: {sorted(final_df['mode'].unique().tolist())}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("FINAL SCORE TABLE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Model                      |          SimpleQA          |          HotpotQA          |          ISAbench         ")
    lines.append(
        "                           |    Base     RAG  Reflexion |    Base     RAG  Reflexion |    Base     RAG  Reflexion"
    )
    lines.append("-----------------------------------------------------------------------------------------------------------------")
    for model in model_order:
        lines.append(
            f"{model_disp[model]:<26} | {score(model, 'simpleqa', 'mode1'):>7} {score(model, 'simpleqa', 'mode2'):>7} {score(model, 'simpleqa', 'mode3'):>10} | "
            f"{score(model, 'hotpotqa', 'mode1'):>7} {score(model, 'hotpotqa', 'mode2'):>7} {score(model, 'hotpotqa', 'mode3'):>10} | "
            f"{score(model, 'qa_iab', 'mode1'):>7} {score(model, 'qa_iab', 'mode2'):>7} {score(model, 'qa_iab', 'mode3'):>10}"
        )

    (out_dir / "evaluation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir / 'detailed_evaluations.csv'}")
    print(f"Wrote {out_dir / 'summary_metrics.csv'}")
    print(f"Wrote {out_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()


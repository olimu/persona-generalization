import json
import argparse
import string
import re
from collections import Counter

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1_single(prediction: str, truth: str) -> float:
    pred_tokens, truth_tokens = normalize_answer(prediction).split(), normalize_answer(truth).split()
    if not pred_tokens or not truth_tokens: return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision, recall = 1.0 * num_same / len(pred_tokens), 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact_match(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))

def judge(preds_file, gt_file, output_file=None):
    with open(preds_file, 'r') as f: preds = [json.loads(l) for l in f if l.strip()]
    with open(gt_file, 'r') as f: gts = [json.loads(l) for l in f if l.strip()]
    
    f1s, ems = [], []
    detailed_records = []

    for p_item, g_item in zip(preds, gts):
        pred = p_item.get("response", p_item.get("text", ""))
        if isinstance(pred, list): pred = pred[0]
        
        targets = []
        gt_msgs = g_item.get("messages", [])
        if len(gt_msgs) > 1 and gt_msgs[1].get("role") == "assistant":
            targets.append(gt_msgs[1]["content"])
        elif "answers" in g_item:
            answers = g_item["answers"]
            targets.extend(answers.get("text", []) if isinstance(answers, dict) else answers)

        max_f1 = max([compute_f1_single(pred, t) for t in targets] or [0.0])
        max_em = max([compute_exact_match(pred, t) for t in targets] or [0.0])
        
        f1s.append(max_f1)
        ems.append(max_em)

        if output_file:
            detailed_records.append({
                "prompt": g_item.get("messages", [{}])[0].get("content", ""),
                "prediction": pred,
                "targets": targets,
                "f1": round(max_f1 * 100, 2),
                "exact_match": round(max_em * 100, 2)
            })

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for rec in detailed_records:
                f_out.write(json.dumps(rec) + "\n")

    return {
        "f1": round((sum(f1s) / len(f1s)) * 100, 2),
        "exact_match": round((sum(ems) / len(ems)) * 100, 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metrics = judge(args.predictions, args.ground_truth, args.output)
    if args.json:
        print(json.dumps(metrics))
    else:
        print(f"F1: {metrics['f1']}% | EM: {metrics['exact_match']}%")

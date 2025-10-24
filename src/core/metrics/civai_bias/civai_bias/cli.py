import argparse
import json 
import sys
from civai_bias.metrics import fact_coverage, opposition_coverage, source_diversity, loaded_intensity
from civai_bias.rules import load_sentiment_lexicon
import os
from transformers.utils.logging import set_verbosity_error
from civai_bias.brief import generate_brief, call_openrouter
from civai_bias.metrics import load_unified_articles, cleanup_temp_files
set_verbosity_error()

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def print_row(row):
    # row = dict with keys: id, fc, oc, sd, li, bias
    print(f"{row['id']:<12}  "
          f"{row['fc']:.2f}   "
          f"{row['oc']:.2f}   "
          f"{row['sd']:.2f}   "
          f"{row['li']:.2f}   "
          f"{row['bias']:.2f}")

def handle_score(article_path, factbank_path, trim):
    weights = load_sentiment_lexicon()

    fc = fact_coverage(article_path, factbank_path)
    oc, L = opposition_coverage(article_path, factbank_path)
    sd = source_diversity(article_path)
    li = loaded_intensity(article_path, weights, trim=trim)

    omission = 1 - (0.6*fc + 0.3*oc + 0.1*sd)
    B = 0.9*omission + 0.1*li
    final_bias = clamp(L * (1 + 0.6*B), -1.0, 1.0)
    
    return {"fc": fc, "oc": oc, "sd": sd, "li": li,"omission": omission, "B": B, "L": L, "bias": final_bias}

def handle_compare(args):
    weights = load_sentiment_lexicon()
    results = []
    for article_path in args.articles:
        fc = fact_coverage(article_path, args.factbank)
        oc, L = opposition_coverage(article_path, args.factbank)
        sd = source_diversity(article_path)
        li = loaded_intensity(article_path, weights, trim=args.trim)

        omission = 1 - (0.6*fc + 0.3*oc + 0.1*sd)
        B = 0.9*omission + 0.1*li
        final_bias = clamp(L * (1 + 0.6*B), -1.0, 1.0)

        try:
            with open(article_path, "r") as f:
                aid = json.load(f).get("id", None)
        except Exception:
            aid = None
        display_id = aid or os.path.basename(article_path)

        row = {
            "id": display_id,
            "fc": fc,
            "oc": oc,
            "sd": sd,
            "li": li,
            "bias": final_bias
        }
        results.append(row)
        print_row(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
parser = argparse.ArgumentParser(prog="bias", description="CivAI bias tools")
subparsers = parser.add_subparsers(dest="command", required=True)

score_p = subparsers.add_parser("score", help="Compute metrics + final bias for one article")
score_p.add_argument("--article", required=True, help="Path to article JSON")
score_p.add_argument("--factbank", required=True, help="Path to factbank JSON")
score_p.add_argument("--trim", type=float, default=0.10, help="Trim ratio for loaded_intensity (0–0.25)")
score_p.add_argument("--json", help="Write results to this JSON file")
score_p.set_defaults(command="score")

compare_p = subparsers.add_parser("compare", help="Compute metrics + final bias for multiple articles")
compare_p.add_argument("--articles", nargs="+", required=True, help="Paths to article JSON files (space-separated)")
compare_p.add_argument("--factbank", required=True, help="Path to factbank JSON")
compare_p.add_argument("--trim", type=float, default=0.10, help="Trim ratio for loaded_intensity (0–0.25)")
compare_p.add_argument("--json", help="If set, write a JSON list of all results to this path")
compare_p.set_defaults(command="compare")

brief_p = subparsers.add_parser("brief", help="Generate bullet lists + optional LLM rewrite from a factbank")
brief_p.add_argument("--factbank", required=True, help="Path to factbank JSON")
brief_p.add_argument("--llm", action="store_true", help="Use OpenRouter to produce a polished brief")
brief_p.add_argument("--json", help="Write result dict to this JSON file")

unified_p = subparsers.add_parser("unified", help="Analyze all articles from unified articles.json")
unified_p.add_argument("--factbank", required=True, help="Path to factbank JSON")
unified_p.add_argument("--trim", type=float, default=0.10, help="Trim ratio for loaded_intensity (0–0.25)")
unified_p.add_argument("--json", help="If set, write a JSON list of all results to this path")
unified_p.set_defaults(command="unified")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.command == "compare":
        handle_compare(args)

    if args.command == "unified":
        # Load articles from unified structure and analyze them
        test_articles = load_unified_articles()
        try:
            # Convert to format expected by handle_compare
            class UnifiedArgs:
                def __init__(self, articles, factbank, trim, json_path):
                    self.articles = articles
                    self.factbank = factbank
                    self.trim = trim
                    self.json = json_path
            
            unified_args = UnifiedArgs(
                articles=list(test_articles.values()),
                factbank=args.factbank,
                trim=args.trim,
                json_path=args.json
            )
            handle_compare(unified_args)
        finally:
            cleanup_temp_files(test_articles)

    if args.command == "score":
        result = handle_score(args.article, args.factbank, args.trim)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)

    if args.command == "brief":
        out = generate_brief(
            factbank_path=args.factbank,
            use_all_facts=True,
            seed=None,
            use_llm=args.llm
        )

        if args.json:
            import json, pathlib
            pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"→ wrote {args.json}")
        else:
            # Pretty print to terminal
            def print_section(title, bullets):
                if bullets:
                    print(title)
                    for b in bullets:
                        print(b)
                    print()  # blank line

            print_section("Facts:", out.get("facts_bullets", []))
            print_section("Supporters say:", out.get("left_bullets", []))
            print_section("Opponents say:", out.get("right_bullets", []))

            if args.llm:
                print("LLM brief:\n")
                print(out.get("llm_brief") or "[LLM unavailable or failed]")
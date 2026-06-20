"""OrganizeMe CLI — local, offline, multimodal file organizer.

Default (no subcommand) launches the interactive flow. Subcommands expose the same
pipeline non-interactively for scripting:

    python main.py organize <dir> --mode content --apply
    python main.py index <dir>
    python main.py search "quarterly revenue" --index <dir>/.organizeme/index
    python main.py dedup <dir> --apply
"""

import os
import sys
import argparse

from organizeme import pipeline, scanning


# --------------------------------------------------------------------------- #
# Interactive flow
# --------------------------------------------------------------------------- #

def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ("yes", "y"):
            return True
        if response in ("no", "n"):
            return False
        if response == "/exit":
            print("Exiting program.")
            sys.exit(0)
        print("Please enter 'yes' or 'no'. To exit, type '/exit'.")


def get_mode_selection():
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content   2. By Date   3. By Type")
        response = input("Enter 1, 2, or 3 (or '/exit'): ").strip()
        if response == "/exit":
            print("Exiting program.")
            sys.exit(0)
        if response in {"1": "content", "2": "date", "3": "type"}:
            return {"1": "content", "2": "date", "3": "type"}[response]
        print("Invalid selection. Please enter 1, 2, or 3.")


def run_interactive():
    while True:
        print("-" * 50)
        input_path = input("Enter the directory you want to organize: ").strip()
        while not os.path.exists(input_path):
            print(f"Input path {input_path!r} does not exist.")
            input_path = input("Enter the directory you want to organize: ").strip()

        output_path = input(
            "Enter the output directory (press Enter for 'organized_folder'): "
        ).strip() or None

        print("\nDirectory tree before organizing:")
        scanning.display_directory_tree(input_path)
        print("*" * 50)

        while True:
            mode = get_mode_selection()
            print(f"Analyzing files (mode={mode}). This may take a few minutes...")
            plan = pipeline.dry_run_organize(input_path, mode, output_path)

            print("-" * 50)
            print("Proposed directory structure:")
            print(os.path.abspath(plan["output_path"]))
            for line in plan["tree"]:
                print(line)
            print("-" * 50)

            if get_yes_no("Proceed with these changes? (yes/no): "):
                pipeline.organize_files(input_path, mode, plan["output_path"], apply=True)
                print("-" * 50)
                print("The files have been organized successfully.")
                print("-" * 50)
                break
            if not get_yes_no("Choose another sorting method? (yes/no): "):
                print("Operation canceled by the user.")
                break

        if not get_yes_no("Organize another directory? (yes/no): "):
            break


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_organize(args):
    plan = pipeline.organize_files(
        args.path, mode=args.mode, output_path=args.output, apply=args.apply
    )
    if not args.apply:
        print(f"\nProposed layout under {os.path.abspath(plan['output_path'])}:")
        for line in pipeline.proposed_tree(plan["operations"], plan["output_path"]):
            print(line)
        print(f"\n{len(plan['operations'])} operation(s). Re-run with --apply to execute.")


def cmd_index(args):
    from organizeme import search
    count = search.build_index(args.path, args.output)
    print(f"Indexed {count} file(s).")


def cmd_search(args):
    from organizeme import search
    results = search.semantic_search(args.query, args.index, k=args.k)
    if not results:
        print("No results.")
        return
    for score, path in results:
        print(f"{score:.3f}  {path}")


def cmd_dedup(args):
    from organizeme import dedup
    report = dedup.detect_duplicates(args.path, threshold=args.threshold, apply=args.apply)
    clusters = report["clusters"]
    if not clusters:
        print("No duplicates found.")
        return
    for cluster in clusters:
        print(f"\nDUPLICATE CLUSTER (method={cluster['method']}, sim={cluster.get('similarity', 1.0):.3f})")
        print(f"  keep: {cluster['keep']}")
        for dup in cluster["duplicates"]:
            print(f"  dupe: {dup}")
    action = "Quarantined" if args.apply else "Would quarantine"
    print(f"\n{action} {report['duplicate_count']} file(s).")


def build_parser():
    parser = argparse.ArgumentParser(description="OrganizeMe — local AI file organizer")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("organize", help="Organize a directory")
    p.add_argument("path")
    p.add_argument("--mode", choices=["content", "date", "type"], default="type")
    p.add_argument("--output", default=None)
    p.add_argument("--apply", action="store_true", help="Execute (default: dry-run)")
    p.set_defaults(func=cmd_organize)

    p = sub.add_parser("index", help="Build a FAISS semantic index for a directory")
    p.add_argument("path")
    p.add_argument("--output", default=None, help="Index dir (default: <path>/.organizeme/index)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="Semantic search over a built index")
    p.add_argument("query")
    p.add_argument("--index", required=True, help="Index dir created by 'index'")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("dedup", help="Detect (and optionally quarantine) duplicates")
    p.add_argument("path")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--apply", action="store_true", help="Quarantine duplicates")
    p.set_defaults(func=cmd_dedup)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        run_interactive()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

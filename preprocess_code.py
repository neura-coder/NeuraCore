import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

def collect_code_files(root_dir: str, extensions: List[str] = [".py", ".js", ".java", ".cpp", ".c"]) -> List[Dict]:
    data = []
    for ext in extensions:
        for path in Path(root_dir).rglob(f"*{ext}"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                if len(code) > 200:  # نادیده گرفتن فایل‌های خیلی کوچک
                    data.append({
                        "code": code,
                        "path": str(path),
                        "language": ext[1:]
                    })
            except:
                continue
    return data

def main():
    parser = argparse.ArgumentParser(description="Preprocess code files into JSON")
    parser.add_argument("--input_dir", required=True, help="Directory containing code files")
    parser.add_argument("--output_json", required=True, help="Output JSON file path")
    parser.add_argument("--extensions", nargs="+", default=[".py"], help="File extensions to include")
    args = parser.parse_args()

    data = collect_code_files(args.input_dir, args.extensions)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} code files to {args.output_json}")

if __name__ == "__main__":
    main()
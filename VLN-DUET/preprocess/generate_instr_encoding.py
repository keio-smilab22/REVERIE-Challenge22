"""
Load json file and add new entry "instr_encodings"
"instr_encodings" should contains lists of integers that represent subword tokens
e.g.
    "instructions": [
        "Go to the dining room on level 1 and open the cabinet drawers",
        "Go to the dining room and open the cabinet doors",
        "go to the dining room on level 1 and open the cabinet below the large painting"
    ],
    "instr_encodings": [
        [101, 2175, 2000, 1996, 7759, 2282, 2006, 2504, 1015, 1998, 2330, 1996, 5239, 22497, 102],
        [101, 2175, 2000, 1996, 7759, 2282, 1998, 2330, 1996, 5239, 4303, 102],
        [101, 2175, 2000, 1996, 7759, 2282, 2006, 2504, 1015, 1998, 2330, 1996, 5239, 2917, 1996, 2312, 4169, 102]
    ]
"""
import argparse
import json
import pathlib
from copy import deepcopy
from typing import Any, Dict, List

from transformers import AutoTokenizer


def get_tokenizer(tokenizer_name: str):
    if tokenizer_name == "xlm":
        cfg_name = "xlm-roberta-base"
    else:
        cfg_name = "bert-base-uncased"
    return AutoTokenizer.from_pretrained(cfg_name)


def main(args: argparse.Namespace):
    tokenizer = get_tokenizer(args.tokenizer)
    input_path = pathlib.Path(args.input).resolve()
    with input_path.open(mode="r") as f:
        info: List[Dict[str, Any]] = json.load(f)

    output_path = deepcopy(input_path)
    input_path.rename(input_path.parent / (input_path.stem + ".old.json"))

    for dic in info:
        instr: List[str] = dic["instructions"]
        instr_encodings: List[int] = []
        for text in instr:
            vocab_ids = tokenizer.encode(text)
            instr_encodings.append(vocab_ids)
        dic["instr_encodings"] = instr_encodings
    with output_path.open(mode="w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=str,
        help="specify input json file. e.g. datasets/REVERIE/annotations/REVERIE_psudo_test_seen.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert",
        help="defalut tokenizer would be bert's wordpiecce (see args listed in `map_nav_src/reverie/scripts/run_reverie.sh`",
    )

    args: argparse.Namespace = parser.parse_args()

    main(args)
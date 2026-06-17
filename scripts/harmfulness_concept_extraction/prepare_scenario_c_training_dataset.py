import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, load_from_disk

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_prediction(pred: str) -> int:
    """
    Parse the evaluator prediction to a binary harmfulness label.
    Return 1 if harmful, 0 if safe.
    """
    pred_lower = pred.strip().lower()

    # Handle WildGuard format explicitly if present
    if "harmful response:" in pred_lower:
        # Extract the value after "harmful response:"
        parts = pred_lower.split("harmful response:")
        if len(parts) > 1:
            val = parts[1].strip().split()[0]
            val = "".join(c for c in val if c.isalpha())
            if "yes" in val:
                return 1
            if "no" in val:
                return 0

    # Llama Guard format (starts with unsafe/safe)
    if pred_lower.startswith("unsafe"):
        return 1
    if pred_lower.startswith("safe"):
        return 0

    # General fallbacks
    if "harmful" in pred_lower:
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare scenario C training dataset with prompt, response, and harmfulness label."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
        help="Path to the original dataset containing prompts.",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        required=True,
        help="Path to the json file containing generated responses.",
    )
    parser.add_argument(
        "--predictions_path",
        type=Path,
        required=True,
        help="Path to the json file containing evaluator predictions (e.g. preds_response.json).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the new HuggingFace dataset.",
    )
    parser.add_argument(
        "--use_cot_only",
        action="store_true",
        help="If set, only the chain of thought part (content inside <think> tags) will be used for the response.",
    )
    args = parser.parse_args()

    logger.info(f"Loading original dataset from {args.dataset_path}")
    dataset = load_from_disk(str(args.dataset_path))

    logger.info(f"Loading generated responses from {args.responses_path}")
    with open(args.responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    logger.info(f"Loading evaluator predictions from {args.predictions_path}")
    with open(args.predictions_path, "r", encoding="utf-8") as f:
        preds_data = json.load(f)
        predictions = preds_data["preds"]

    # Ensure lengths match
    min_len = min(len(dataset), len(responses), len(predictions))
    if not (len(dataset) == len(responses) == len(predictions)):
        logger.warning(
            f"Length mismatch! Dataset: {len(dataset)}, Responses: {len(responses)}, Predictions: {len(predictions)}. Truncating to {min_len}."
        )

    new_data = {
        "messages": [],
        "labels": [],
        # "evaluator_prediction": [],
    }

    dropped_no_thinking = 0

    for i in range(min_len):
        example = dataset[i]
        full_response = responses[i]
        pred = predictions[i]

        if "</think>" not in full_response:
            dropped_no_thinking += 1
            continue

        # Extract prompt from original dataset
        user_content = ""
        if "prompt" in example:
            user_content = example["prompt"]
        elif "instruction" in example:
            user_content = example["instruction"]
        elif "messages" in example:
            for msg in example["messages"]:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

        if not user_content:
            user_content = "N/A"

        # Extract only the chain of thought if flag is set
        if args.use_cot_only:
            parts = full_response.split("</think>", 1)
            if len(parts) > 1:
                if not parts[0].startswith("<think>"):
                    parts[0] = "<think>" + parts[0]
                response_content = parts[0] + "</think>"
            else:
                response_content = full_response
        else:
            response_content = full_response

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response_content},
        ]

        label = parse_prediction(pred)

        new_data["messages"].append(messages)
        new_data["labels"].append(label)
        # new_data["evaluator_prediction"].append(pred) # Keep raw prediction text for debugging

    logger.info(f"Dropped {dropped_no_thinking} examples due to missing thinking tags.")
    logger.info(f"Creating new dataset with {len(new_data['messages'])} examples.")
    new_dataset = Dataset.from_dict(new_data)

    logger.info(f"Saving dataset to {args.output_path}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # summarize dataset
    logger.info(f"Dataset summary: {new_dataset}")
    # num of harmful and safe examples
    logger.info(f"Number of harmful examples: {new_dataset['labels'].count(1)}")
    logger.info(f"Number of safe examples: {new_dataset['labels'].count(0)}")
    new_dataset.save_to_disk(str(args.output_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()

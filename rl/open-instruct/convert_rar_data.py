import json
from datasets import load_dataset, Dataset
import argparse

# RUBRIC_SUFFIX = "Output a score of either 1 or 0, depending on whether the response satisfies the criteria."
def convert_rar_data(orig_dataset_id, new_dataset_id):
    original_dataset = load_dataset(orig_dataset_id, split="train")
    dataset_name = new_dataset_id.split("/")[-1]
    
    print("Dataset info:")
    print(f"Number of examples: {len(original_dataset)}")
    print(f"Features: {original_dataset.features}")
    print("\nColumn names:")
    print(original_dataset.column_names)

    new_dataset = []
    for example in original_dataset:
        original_rubrics = example["rubric"]
        for rubric in original_rubrics:
            assert "description" in rubric, rubric
            rubric["description"] = rubric["description"]
        new_dataset.append({
            "source": dataset_name,
            "question_type": "long_form",
            "messages": [{"content": example["question"], "role": "user"}],
            "ground_truth": json.dumps({
                "query": example["question"],
                "rubrics": original_rubrics,
            }),
            "dataset": "general_rubric"
        })

    modified_dataset = Dataset.from_list(new_dataset)
    print(f"\nModified dataset columns: {modified_dataset.column_names}")
    print(f"First example with new column:")
    print({k: v for k, v in modified_dataset[0].items() if k != 'ground_truth'})

    # upload to new dataset
    modified_dataset.push_to_hub(new_dataset_id, private=True, split="train")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_org", type=str, required=True)
    args = parser.parse_args()
    
    convert_rar_data("anisha2102/RaR-Medicine-20k-o3-mini", f"{args.hf_org}/RaR-Medicine-20k-o3-mini-converted")
    print(f"Uploaded to {args.hf_org}/RaR-Medicine-20k-o3-mini-converted")
    convert_rar_data("anisha2102/RaR-Science-20k-o3-mini", f"{args.hf_org}/RaR-Science-20k-o3-mini-converted")
    print(f"Uploaded to {args.hf_org}/RaR-Science-20k-o3-mini-converted")

if __name__ == "__main__":
    main()

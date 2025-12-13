from collections import defaultdict
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..tokenizer_utils import (
    decode_with_offset_mapping,
    find_tag_spans_in_text
)
from .supervised import SupervisedDatasetProcessor

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)



@dataclass
class SpanMaskedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    """
    A dataset processor that extends SupervisedDatasetProcessor to support span-based token masking.

    """

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        """
        Extended version of _encode_data_example that supports span masking.
        """
        # First, get the standard encoding from the parent class
        input_ids, labels = super()._encode_data_example(
            prompt, response, system, tools, images, videos, audios
        )

        assert len(labels) == len(input_ids)

        if self.data_args.mask_span_types is not None:
            # 1. Decode back the input_ids with offset mapping
            # print("mask_span_types: ", self.data_args.mask_span_types)
            original_text = decode_with_offset_mapping(input_ids, self.tokenizer)
            # print("tag in original_text: ", any(tag in original_text['text'] for tag in self.data_args.mask_span_types))

            # 2. Find the spans of the span_tags
            all_mask_spans = []
            for tag in self.data_args.mask_span_types:
                spans = find_tag_spans_in_text(original_text['text'], tag)
                all_mask_spans.extend(spans)

            # 3. Then we use the string_index_to_token_index to get the token spans
            token_spans = []
            for span in all_mask_spans:
                current_token_spans = original_text["string_index_to_token_index"][span[0]:span[1]]
                min_token_span = min([ele[2] for ele in current_token_spans])
                max_token_span = max([ele[2]+1 for ele in current_token_spans])
                token_spans.append((min_token_span, max_token_span))
            
            # 3.1: Optional: 
            # preprocess such that it will remove the example in the user message span 

            # 4. Then we mask the labels
            for token_span in token_spans:
                labels[token_span[0]:token_span[1]] = [IGNORE_INDEX] * (token_span[1] - token_span[0])

            # print("input_ids: ", input_ids)
            # print("labels: ", labels)
            # print("token_spans: ", token_spans)
            # sys.exit()

        return input_ids, labels


    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        Print a data example with span masking information.
        """
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        masked_count = len(example["labels"]) - len(valid_labels)

        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        print(f"masked_tokens_count: {masked_count}")

        # Show which tokens are masked
        masked_positions = [
            i for i, label in enumerate(example["labels"]) if label == IGNORE_INDEX
        ]
        if masked_positions:
            print(f"masked_positions: {masked_positions}")

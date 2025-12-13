import json
import os
import re
from collections import defaultdict
from typing import Dict, Any, Optional, List
import argparse

def extract_citations_from_context(context: str) -> Dict[str, str]:
    """
    Extract citations from the context.

    Citations are expected to be in the format:
    <snippets id="a1b2c3d4" metadata='{"author": "smith", "source": "arxiv", "year": 2023}'>
        Search result content here
    </snippets>
    
    Other formats are also supported, such as:
    <snippet id="a1b2c3d4">Search result content here</snippet>
    <webpage id="a1b2c3d4">Search result content here</webpage>
    
    The id can be any string, including hash-like IDs (e.g., a1b2c3d4)

    Args:
        context: The context string containing citations

    Returns:
        Dictionary mapping id to search results content for all found citations
    """
    citations = defaultdict(str)

    # Pattern to match <snippets id="xxx" ...> content </snippets>
    # Updated to handle both quoted and unquoted attributes in HTML-like format
    pattern1 = r'<snippets?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets?>'
    pattern2 = r'<snippet?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippet?>'
    pattern3 = r'<webpage?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</webpage?>'

    matches1 = re.findall(pattern1, context, re.DOTALL)
    matches2 = re.findall(pattern2, context, re.DOTALL)
    matches3 = re.findall(pattern3, context, re.DOTALL)

    for quote_char, snippet_id, search_results in matches1 + matches2 + matches3:
        # Clean up the id and search results (remove extra whitespace)
        clean_id = snippet_id.strip()
        clean_search_results = search_results.strip()
        citations[clean_id] = clean_search_results

    return citations

def change_citations_astabench_format(text: str) -> str:
    """
    Change citations in the text from <cite id="xxx"></cite> to [id]

    Args:
        text: The text containing citations

    Returns:
        The text with citations changed to [CITATION xxx] format
    """
    citation_sub_pattern = r'[ \t]*<cite id=["\']?([^"\'>\s]+)["\']?>\s*(.*?)\s*</cite>[ \t]*'

    citation_ids = []

    def repl(match):
        ids = match.group(1).split(",")   # split multi-cite ids
        snippet = match.group(2).strip()

        citation_ids.extend(ids)
        cites = "".join(f"[{cid}]" for cid in ids)

        # If snippet ends with sentence punctuation, insert cites before it
        if snippet and snippet[-1] in ".!?,":
            return " " + snippet[:-1] + " " + cites + snippet[-1] + " "
        else:
            return " " + snippet + " " + cites + " "

    # handles weird cases when LLM doesn't follow the format exactly
    new_text = re.sub(citation_sub_pattern, repl, text)
    new_text = new_text.replace("  ", " ").strip()
    new_text = new_text.replace(" .", ".")

    return new_text, citation_ids

def extract_title_snippet(text: str) -> Optional[tuple[str, str]]:
    pattern = r'^Title:\s*(.*?)\nSnippet:\s*(.*)$'
    match = re.match(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return None, text

# fixed version!!

def parse_answer(sample: str) -> List[Dict[str, Any]]:
    sections = []

    # merge section headers and text
    raw_sections_temp = sample["final_response"].strip().split("\n\n")
    raw_sections = []
    current = None
    for line in raw_sections_temp:
        if line.startswith("#"):
            if current is not None:
                raw_sections.append(current)
            current = line
        else:
            if current is not None:
                current += "\n\n" + line
            else:
                raw_sections.append(line)
    # last heading
    if current is not None:
        raw_sections.append(current)

    # remove sections that are just a single heading line
    raw_sections = [
        sec for sec in raw_sections
        if not (sec.strip()[0] == '#' and len(sec.strip().split('\n')) == 1)
    ]

    # get all snippets from the thinking section
    if '<think>' in sample['full_traces']['generated_text']:
        snippets = extract_citations_from_context(sample['full_traces']['generated_text'])
    else:
        concat_text = "" 
        for tool_call in sample['full_traces']['tool_calls']:
            concat_text += tool_call['generated_text'] + "\n"
        snippets = extract_citations_from_context(concat_text)

    for section in raw_sections:
        title_match = re.match(r"#+\s*([^\n]*)", section)
        if title_match:
            title = "# " + title_match.group(1).strip()
            text_body = section[title_match.end():].strip()
        else:
            title = None
            text_body = section.strip()

        text_clean, citation_ids = change_citations_astabench_format(text_body)

        section_citations = []
        for cit_id in set(citation_ids):
            cit_title, cit_snippet = extract_title_snippet(snippets[cit_id])
            section_citations.append({"id": f'[{cit_id}]', 'title': cit_title, "snippets": [cit_snippet]})

        sections.append({
            "title": title,
            "text": text_clean,
            "citations": section_citations
        })

    return sections

parser = argparse.ArgumentParser(description="Convert files to ASTA format")
parser.add_argument('--folder', type=str, required=True, help='Path to the folder where eval file is stored')
parser.add_argument('--file', type=str, required=True, help='Name of the eval file')
args = parser.parse_args()
folder = args.folder
file = args.file

print(f"Processing file: {file}")
# read data
file_path = os.path.join(folder, file)
data = []

with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

formatted = []
for sample in data:
    parsed = parse_answer(sample)
    formatted.append({"question": sample['problem'], "response": {"sections": parsed}})


# write formatted data to a json file
out_file = file_path.replace(".jsonl", "_asta_format.jsonl")
with open(out_file, "w") as f:
    json.dump(formatted, f, indent=4)

import json
import os
import re
import copy
from typing import List, Dict, Any
import time

import re
import argparse

def parse_xml_snippets(text):
    """
    Parses a string of XML-like snippets into a dictionary.

    The output is a dictionary where keys are snippet IDs and values are
    dictionaries containing the Title, URL, and Snippet.

    Args:
        text: The input string containing one or more <snippet> blocks.

    Returns:
        A dictionary mapping each snippet ID to its parsed content.
        Example: {'e08c9291-0': {'Title': '...', 'URL': '...', 'Snippet': '...'}}
    """
    # Regex to capture the ID from the tag, the Title, an optional URL,
    # and the Snippet content from within the tags.
    # Key changes:
    # 1. `<snippet id=([^>]+)>`: Captures the ID (anything not a '>') from the opening tag.
    # 2. `\s*`: Matches any whitespace (including newlines) between fields for robustness.
    # 3. `</snippet>`: Marks the end of the block.
    # re.DOTALL is crucial for the Snippet content to span multiple lines.
    pattern = re.compile(
        r"<snippet id=([^>]+)>"  # Capture Group 1: The ID
        r"\s*Title: (.*?)"         # Capture Group 2: The Title
        r"\s*(?:URL: (.*?))?"      # Optional Group with Capture Group 3: The URL
        r"\s*Snippet: (.*?)"       # Capture Group 4: The Snippet
        r"\s*</snippet>",          # The closing tag
        re.DOTALL
    )

    matches = pattern.findall(text)

    results = {}
    for match in matches:
        # match is a tuple: (id, title, url, snippet)
        # If the optional URL group did not match, match[2] will be None.
        snippet_id = match[0].strip()
        title = match[1].strip()
        url = match[2].strip() if match[2] else ""  # Handle None case for missing URL
        snippet = match[3].strip()

        results[snippet_id] = {
            "Title": title,
            "URL": url,
            "Snippet": snippet
        }

    return results

def parse_and_format_citations(text):
    """
    Parses a string with custom <cite> tags, extracts citation IDs,
    and reformats the string.

    Args:
        text: The input string containing <cite> tags.

    Returns:
        A tuple containing:
        - The formatted string.
        - A list of citation IDs in the order of their appearance.
    """
    citation_ids = []
    id_to_text_map = {}  # Initialize the new list for the mapping


    def replacer(match):
        """
        This function is called for each match found by re.sub().
        """
        # Extract the comma-separated ids and the text from the matched groups
        ids_string = match.group(1)
        text_content = match.group(2)

        # Split the ids string by comma and add them to our list
        ids = ids_string.split(',')

        for this_id in ids:
            if this_id not in citation_ids:
                citation_ids.append(this_id)

        for cid in ids:
            if cid not in id_to_text_map:
                id_to_text_map[cid] = [text_content]
            else:
                id_to_text_map[cid].append(text_content)

        # Format the citation ids part of the replacement string
        formatted_ids = ' '.join([f'[{id}]' for id in ids])

        # Return the desired formatted string
        return f". {text_content} {formatted_ids}."

    # The regular expression to find the cite tags
    # It captures the id attribute value in group 1 and the tag's text content in group 2
    pattern = re.compile(r'<cite id="([^"]+)">([^<]+)</cite>')
    
    formatted_text = pattern.sub(replacer, text)
    
    return formatted_text, citation_ids, id_to_text_map


def parse_search_results(text):
    """
    Parses a string of search results into a list of dictionaries.
    Handles entries with and without a URL line.

    Args:
        text: The input string containing one or more search results.

    Returns:
        A list of dictionaries, where each dictionary represents a search result
        with 'Title', 'URL', and 'Snippet' keys. If a URL is not found,
        it is set to an empty string.
    """
    # Regex to capture Title, an optional URL, and Snippet.
    # The URL line is wrapped in a non-capturing group (?:...) and made optional with '?'
    # This ensures that the group doesn't create an extra capture index.
    # The inner group for the URL itself `(.*?)` remains a capturing group.
    # re.DOTALL allows '.' to match newlines, crucial for the snippet.
    pattern = re.compile(
        r"Title: (.*?)\n"           # Capture Group 1: Title
        r"(?:URL: (.*?)\n)?"       # Optional Non-capturing group for the whole line
                                    # with a Capturing Group 2 for the URL value
        r"Snippet: (.*?)"           # Capture Group 3: Snippet
        r"(?=\nTitle: |\Z)",        # Lookahead for the next Title or end of string
        re.DOTALL
    )
    
    matches = pattern.findall(text)
    
    results = []
    for match in matches:
        # The match tuple will have 3 elements: (title, url, snippet)
        # If the optional URL group did not match, match[1] will be an empty string.
        results.append({
            "Title": match[0].strip(),
            "URL": match[1].strip(),
            "Snippet": match[2].strip()
        })
        
    return results

def contains_chinese_regex(text):
    """
    Checks if a string contains any Chinese characters using regex.
    """
    # This regex pattern will match any string that contains at least one character
    # in the CJK Unified Ideographs range.
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def data_format(example):
    output = {}
    if "example_id" in example:
        output["id"] = example["example_id"]
    else:
        output["id"] = example["id"]
    output["prompt"] = example["problem"]
    output["article"] = ""
    output["citations_deduped"] = {}
    output["citations"] = []
    
    traces = {}
    all_urls = {}
    if "full_traces" in example:    
        for call in example["full_traces"]["tool_calls"]:
            if "call_id" in call:
                call_id = call["call_id"]

                call_results = parse_search_results(call["output"])
                if len(call_results) ==0:
                    try:
                        aggregated_results = "\n\n".join(raw_call["output"].strip() for raw_call in call["raw_output"]['tool_outputs'])
                        call_results = parse_search_results(aggregated_results)
                    except:
                        # Exceed allowed tool call requests.
                        action = "skip"
                        # print(call.keys())
                        # print(call["raw_output"])

                for i, each_result in enumerate(call_results):
                    traces[call_id + "-" + str(i)] = each_result
                    all_urls[each_result["URL"]] = call_id + "-" + str(i)
            else:
                # call id is not parsed, and in the generated text
                snippets = parse_xml_snippets(call["generated_text"])
                # print(snippets)
                for snippet_id, snippet_content in snippets.items():
                    traces[snippet_id] = snippet_content
                    all_urls[snippet_content["URL"]] = snippet_id
                # prin

    # cite format: <cite id=\"cd29c481-1\">xxx</cite>
    if "final_response" not in example:
        if "pred_answer" in example:
            example["final_response"] = example["pred_answer"]

    article, citation_ids, id_to_text_map = parse_and_format_citations(example["final_response"])

    for url, cid in all_urls.items():
        if cid in id_to_text_map:
            output["citations_deduped"][url] = {
                "facts": id_to_text_map[cid],
                "url_content": traces[cid]["Title"] + "\n\n" + traces[cid]["Snippet"]
            }
            for fact in id_to_text_map[cid]:
                output["citations"].append({
                    "fact": fact,
                    "ref_indx": cid,
                    "url": url
                })
    # {fact, ref_indx, url}

    reference = []
    for i, each_id in enumerate(citation_ids):
        if each_id in traces:
            reference.append("[" + str(i+1) + "] " + traces[each_id]["URL"])
            article = article.replace(each_id, str(i+1))
        else:

            indicator = False
            for each_url in all_urls:
                if each_id in each_url:
                    reference.append("[" + str(i+1) + "] " + each_url)
                    article = article.replace(each_id, str(i+1))
                    indicator = True
                    break
            if not indicator:
                article = article.replace("[" + each_id + "]", "")
    
    if contains_chinese_regex(article):
        output["article"] = article + "\n\n 参考文献: " + "\n".join(reference)
    else:
        output["article"] = article + "\n\n References: " + "\n".join(reference)

    return output




if __name__ == "__main__":
    # two args: input_file and output_file_name
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--drb_repo_path", type=str, required=True)
    args = parser.parse_args()

    data = []
    with open(args.input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    export_folder = f"{args.drb_repo_path}/data/test_data/raw_data/"
    
    # if task name JSONL already exists, print a warning and add a timestamp to the file name
    if os.path.exists(os.path.join(export_folder, args.task_name + ".jsonl")):
        print(f"Warning: {args.task_name}.jsonl already exists. Adding timestamp to the file name.")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.task_name = args.task_name + "_" + timestamp
        print(f"New task name: {args.task_name}")

    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    formated_data = []
    for i, item in enumerate(data):
        try:
            formated_data.append(data_format(item))
        except:
            print("Error in formatting data point: ", i)

    # dump formated data to jsonl
    with open(os.path.join(export_folder, args.task_name + ".jsonl"), 'w', encoding='utf-8') as f:
        for item in formated_data:
            f.write(json.dumps(item) + "\n")

    # dump scraped data to jsonl
    scrape_folder = f"{args.drb_repo_path}/results/fact/{args.task_name}/"
    if not os.path.exists(scrape_folder):
        os.makedirs(scrape_folder)
    with open(os.path.join(scrape_folder, "scraped.jsonl"), 'w', encoding='utf-8') as f:
        for item in formated_data:
            f.write(json.dumps(item) + "\n")


    print(f"Formated data saved to {export_folder}{args.task_name}.jsonl")


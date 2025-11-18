import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from requests.exceptions import RequestException

from ..cache import cached
from .semantic_scholar_apis import (
    download_paper_details_batch as download_paper_details_from_semantic_scholar_batch,
)
from .utils import call_api_with_retry

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def extract_all_text(tag: ElementTree.Element) -> str:
    """
    Sometimes tag.text will produce a None value if there's rich text
    inside the tag. For example,

    In this paper, https://pubmed.ncbi.nlm.nih.gov/39355906/, the returned
    title data is the following:

    <ArticleTitle><i>LRP1</i> Repression by SNAIL Results in ECM Remodeling
    in Genetic Risk for Vascular Diseases.</ArticleTitle>

    And tag.text will return None.

    This function will extract all text from the tag, including rich text.
    """
    return " ".join([_.strip() for _ in tag.itertext()])


def search_pubmed_with_keywords(keywords: str, offset: int = 0, limit: int = 10):
    search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": keywords,
        "retmax": limit,
        "retstart": offset,
        "usehistory": "n",
        "sort": "relevance",
        "email": "your_email@example.com",
    }
    response = requests.get(search_url, params=params)
    root = ElementTree.fromstring(response.content)
    id_list = [id_elem.text for id_elem in root.findall("./IdList/Id")]
    #  (root.find("./Count").text, root.find("./RetStart").text, root.find("./RetMax").text)
    return {
        "ids": id_list,
        "count": root.find("./Count").text,
        "offset": root.find("./RetStart").text,
        "limit": root.find("./RetMax").text,
        "next": int(root.find("./RetStart").text) + int(root.find("./RetMax").text),
    }


def fetch_pubmed_details(id_list):
    fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "email": "your_email@example.com",
    }
    response = requests.get(fetch_url, params=params)
    papers = ElementTree.fromstring(response.content)

    paper_data_list = []
    for paper in papers.findall("./PubmedArticle"):
        article = paper.find(".//Article")
        pmid = paper.find(".//PMID").text
        title = (
            extract_all_text(article.find(".//ArticleTitle"))
            if article.find(".//ArticleTitle") is not None
            else ""
        )
        abstract = (
            "\n".join(
                [
                    extract_all_text(abstract_text)
                    for abstract_text in article.findall(".//Abstract/AbstractText")
                ]
            )
            if article.find(".//Abstract") is not None
            else None
        )
        abstract = []
        if article.find(".//Abstract") is not None:
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                if abstract_text.attrib.get("Label"):
                    abstract.append(f"{abstract_text.attrib['Label']}")
                abstract.append(extract_all_text(abstract_text))
        abstract = "\n".join(abstract)

        authors = [
            {
                "name": f"{author.find('./LastName').text}, {author.find('./ForeName').text}"
            }
            for author in article.findall(".//Author")
            if author.find("./LastName") is not None
            and author.find("./ForeName") is not None
        ]
        year = article.find(".//Journal/JournalIssue/PubDate/Year")
        venue = (
            article.find(".//Journal/Title").text
            if article.find(".//Journal/Title") is not None
            else None
        )
        article_dates = article.findall(".//ArticleDate")

        publication_date = None
        if article_dates:
            # Grab the first ArticleDate's Year element. Adjust as necessary for Month/Day.
            publication_date = article_dates[0].find("Year").text

        paper_data = {
            "paperId": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "externalIds": {"PubMed": pmid},
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year.text if year is not None else None,
            "venue": venue,
            "publicationDate": publication_date,
        }
        paper_data_list.append(paper_data)
    return paper_data_list


def fetch_semantic_scholar_details(paper_data: List[Dict]):

    paper_ids = [f'PMID:{paper["externalIds"]["PubMed"]}' for paper in paper_data]

    try:
        results = download_paper_details_from_semantic_scholar_batch(paper_ids)
        # print(results)
        # print(paper_data)
        for idx in range(len(paper_data)):
            semantic_scholar_data = results[idx]
            for key in semantic_scholar_data.keys():
                if key not in paper_data[idx]:
                    paper_data[idx][key] = semantic_scholar_data[key]
        # print(paper_data)
    except:
        for paper in paper_data:
            paper.update({"citationCount": None})
        print("Error fetching data from Semantic Scholar")

    # for paper in paper_data:
    #     paper_id = paper["externalIds"]["PubMed"]
    #     try:
    #         semantic_scholar_data = download_paper_details_from_semantic_scholar(f"PMID:{paper_id}")
    #         # We prioritize the data from PubMed
    #         for key in semantic_scholar_data.keys():
    #             if key not in paper:
    #                 paper[key] = semantic_scholar_data[key]
    #     except:
    #         paper.update({"citationCount": None})
    #     time.sleep(0.2)  # Add a delay to avoid rate limiting

    return paper_data


def search_pubmed(
    keywords: str,
    limit: int = 10,
    offset: int = 0,
):
    searchStat = search_pubmed_with_keywords(keywords, offset=offset, limit=limit)
    ids = searchStat["ids"]

    paper_data = fetch_pubmed_details(ids)
    paper_data = call_api_with_retry(fetch_semantic_scholar_details, paper_data)

    return {
        "total": int(searchStat["count"]),
        "offset": int(searchStat["offset"]),
        "next": int(searchStat["next"]),
        "data": paper_data,
    }

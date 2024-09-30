import requests
import time
import json
import os
import concurrent.futures
import logging
from typing import List, Dict, Any, Tuple
import google.generativeai as genai 
import typing_extensions as typing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL
BASE_URL = "https://www.courtlistener.com/api/rest/v4"

class CitationAnalysis(typing.TypedDict):
    cited_case_name: str
    cited_case_citation: str
    citing_case_name: str
    citing_case_citation: str
    label: str
    classification: str
    reasoning: str

def make_request(url: str, headers: Dict[str, str], max_retries: int = 5, initial_wait: int = 5) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_time = initial_wait * (2 ** attempt)
                logging.warning(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Error: Status code {response.status_code} for URL: {url}")
                return None
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
    
    logging.error(f"Max retries reached for URL: {url}")
    return None

def get_case_name(opinion_id: str, headers: Dict[str, str]) -> str:
    url = f"{BASE_URL}/clusters/{opinion_id}/"
    data = make_request(url, headers)
    if data:
        return data.get('case_name') or data.get('case_name_full') or "Unknown Case Name"
    return "Unknown Case Name"

def get_citing_opinions(opinion_id: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/search/?q=cites%3A({opinion_id})"
    data = make_request(url, headers)
    if data:
        return data.get('results', [])[:3]  # Limit to first three opinions
    return []

def process_single_opinion(main_case_name: str, citing_case_name: str, date: str, opinion_text: str, genai_model) -> Dict[str, Any]:
    prompt = f"""Analyze the following opinion text and extract information about how it cites and treats the case "{main_case_name}". 
    Provide the output in the following JSON format:

    {{
        "cited_case_name": "The name of the main case being cited",
        "cited_case_citation": "The legal citation for the main case",
        "citing_case_name": "{citing_case_name}",
        "citing_case_citation": "The legal citation for the citing case",
        "label": "A brief label describing how the citing case treats the cited case. Here are the main characteristics of the labels:
        - 'followed': The citing case followed the cited case as precedent.
        - 'distinguished': The citing case distinguished the cited case, but did not follow it.
        - 'partially overruled': The citing case partially overruled the cited case.
        - 'overruled': The citing case overruled the cited case.
        - 'rejected': The citing case rejected the cited case as precedent.
        - 'mentioned': The citing case mentioned the cited case, but did not treat it as precedent.
        - 'rejected': The citing case rejected the cited case as precedent.
        "reasoning": "A brief explanation of why this classification was chosen, based on the content of the opinion"
    }}

    Opinion Text:
    {opinion_text}

    Ensure that all fields are filled out based on the information available in the opinion text. If any information is not available, use "Unknown" as the value.
    """

    try:
        response = genai_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=CitationAnalysis
            )
        )
        
        result = json.loads(response.text)
        return result
    except Exception as e:
        logging.error(f"Error processing opinion {citing_case_name}: {str(e)}")
        return None

def process_opinion_worker(main_case_name: str, opinion: Dict[str, Any], headers: Dict[str, str], genai_model) -> Dict[str, Any]:
    citing_case_name = opinion.get('caseName') or opinion.get('caseNameFull', 'Unknown Case Name')
    date_filed = opinion.get('dateFiled', 'Unknown Date')
    
    content = None
    if opinion.get('opinions'):
        first_opinion = opinion['opinions'][0]
        opinion_id = first_opinion.get('id')
        if opinion_id:
            opinion_url = f"{BASE_URL}/opinions/{opinion_id}/"
            logging.info(f"Fetching full opinion data from: {opinion_url}")
            opinion_data = make_request(opinion_url, headers)
            if opinion_data:
                content = opinion_data.get('plain_text') or opinion_data.get('html') or \
                        opinion_data.get('html_lawbox') or opinion_data.get('html_columbia') or \
                        opinion_data.get('html_anon_2020') or opinion_data.get('xml_harvard') or \
                        opinion_data.get('html_with_citations')
    
    if content:
        processed_result = process_single_opinion(main_case_name, citing_case_name, date_filed, content, genai_model)
        if processed_result:
            logging.info(f"Successfully processed citing opinion: {citing_case_name}")
            return processed_result
        else:
            logging.warning(f"Failed to process citing opinion: {citing_case_name}")
            return None
    else:
        logging.warning(f"No content available for citing opinion: {citing_case_name}")
        return None

def process_opinion(opinion_id: str, headers: Dict[str, str], genai_model) -> Tuple[str, List[Dict[str, Any]]]:
    main_case_name = get_case_name(opinion_id, headers)
    logging.info(f"Main Case Name for opinion {opinion_id}: {main_case_name}")
    
    logging.info(f"Fetching citing opinions for opinion ID: {opinion_id}")
    citing_opinions = get_citing_opinions(opinion_id, headers)
    
    logging.info(f"Processing the first {len(citing_opinions)} citing opinions in parallel...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_opinion = {executor.submit(process_opinion_worker, main_case_name, opinion, headers, genai_model): opinion for opinion in citing_opinions}
        for future in concurrent.futures.as_completed(future_to_opinion):
            result = future.result()
            if result:
                results.append(result)
    
    return main_case_name, results

def save_results_to_file(main_case_name: str, results: List[Dict[str, Any]], filename: str) -> None:
    output = {
        "main_case_name": main_case_name,
        "citing_opinions": results
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

def main():
    print("Welcome to the Opinion Citation Analyzer")
    
    auth_token = input("Please enter your Court Listener AUTH_TOKEN: ").strip()
    genai_api_key = input("Please enter your Google Generative AI API Key: ").strip()
    opinion_id = input("Please enter the Opinion ID you want to analyze: ").strip()

    headers = {
        'Authorization': f'Token {auth_token}'
    }

    genai.configure(api_key=genai_api_key)
    genai_model = genai.GenerativeModel("gemini-1.5-pro-latest")

    print(f"\nAnalyzing citations for Opinion ID: {opinion_id}")
    main_case_name, results = process_opinion(opinion_id, headers, genai_model)

    print(f"\nMain Case: {main_case_name}")
    print(f"Number of citing opinions processed: {len(results)}")

    if results:
        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('citing_case_name', 'Unknown Case')}")
            print(f"   Treatment: {result.get('label', 'Unknown')}")
            print(f"   Reasoning: {result.get('reasoning', 'No reasoning provided')[:100]}...")  # Truncate reasoning for display

        output_filename = f'processed_opinions_{opinion_id}.json'
        save_results_to_file(main_case_name, results, output_filename)
        print(f"\nFull results saved to: {output_filename}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()

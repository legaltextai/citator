import streamlit as st
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
st.title ("Opinion Citation Analyzer") 


# Environment variables
BASE_URL = os.getenv('BASE_URL', "https://www.courtlistener.com/api/rest/v4")
AUTH_TOKEN = os.getenv('AUTH_TOKEN', "your_courtlistener_authorization_token")
GENAI_API_KEY = os.getenv('GENAI_API_KEY', "google_gemini_api")

HEADERS = {
    'Authorization': f'Token {AUTH_TOKEN}'
}

genai.configure(api_key=GENAI_API_KEY)

class CitationAnalysis(typing.TypedDict):
    cited_case_name: str
    cited_case_citation: str
    citing_case_name: str
    citing_case_citation: str
    label: str
    reasoning: str

def make_request(url: str, max_retries: int = 5, initial_wait: int = 5) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS)
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

def get_case_name(opinion_id: str) -> str:
    url = f"{BASE_URL}/clusters/{opinion_id}/"
    data = make_request(url)
    if data:
        return data.get('case_name') or data.get('case_name_full') or "Unknown Case Name"
    return "Unknown Case Name"

def get_citing_opinions(opinion_id: str) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/search/?q=cites%3A({opinion_id})"
    data = make_request(url)
    if data:
        return data.get('results', [])[:3]  # Limit to first opinion
    return []

def process_single_opinion(main_case_name: str, citing_case_name: str, date: str, opinion_text: str) -> Dict[str, Any]:
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

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    try:
        response = model.generate_content(
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

def process_opinion_worker(main_case_name: str, opinion: Dict[str, Any]) -> Dict[str, Any]:
    citing_case_name = opinion.get('caseName') or opinion.get('caseNameFull', 'Unknown Case Name')
    date_filed = opinion.get('dateFiled', 'Unknown Date')
    
    content = None
    if opinion.get('opinions'):
        first_opinion = opinion['opinions'][0]
        opinion_id = first_opinion.get('id')
        if opinion_id:
            opinion_url = f"{BASE_URL}/opinions/{opinion_id}/"
            logging.info(f"Fetching full opinion data from: {opinion_url}")
            opinion_data = make_request(opinion_url)
            if opinion_data:
                content = opinion_data.get('plain_text') or opinion_data.get('html') or \
                        opinion_data.get('html_lawbox') or opinion_data.get('html_columbia') or \
                        opinion_data.get('html_anon_2020') or opinion_data.get('xml_harvard') or \
                        opinion_data.get('html_with_citations')
    
    if content:
        processed_result = process_single_opinion(main_case_name, citing_case_name, date_filed, content)
        if processed_result:
            logging.info(f"Successfully processed citing opinion: {citing_case_name}")
            return processed_result
        else:
            logging.warning(f"Failed to process citing opinion: {citing_case_name}")
            return None
    else:
        logging.warning(f"No content available for citing opinion: {citing_case_name}")
        return None

def process_opinion(opinion_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    main_case_name = get_case_name(opinion_id)
    logging.info(f"Main Case Name for opinion {opinion_id}: {main_case_name}")
    
    logging.info(f"Fetching citing opinions for opinion ID: {opinion_id}")
    citing_opinions = get_citing_opinions(opinion_id)
    
    logging.info(f"Processing the first {len(citing_opinions)} citing opinions in parallel...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_opinion = {executor.submit(process_opinion_worker, main_case_name, opinion): opinion for opinion in citing_opinions}
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


opinion_id = st.text_input("Enter Opinion ID:", "2673149")

if st.button("Analyze Citations"):
    with st.spinner("Fetching main case name..."):
        main_case_name = get_case_name(opinion_id)
        st.write(f"Main Case: {main_case_name}")

    with st.spinner("Processing citing opinions..."):
        main_case_name, results = process_opinion(opinion_id)

    if results:
        
        st.success(f"Successfully processed {len(results)} citing opinions for {main_case_name}")
        for i, result in enumerate(results, 1):
            with st.expander(f"{i}. {result.get('citing_case_name', 'Unknown Case')}"):
                st.json(result)
    else:
        st.warning(f"No results found for {main_case_name}")

    output_filename = f'processed_opinions_{opinion_id}.json'
    save_results_to_file(main_case_name, results, output_filename)
    st.info(f"Processed results have been saved to '{output_filename}'")

    # Provide download link for the JSON file
    with open(output_filename, "rb") as file:
        st.download_button(
            label="Download JSON Results",
            data=file,
            file_name=output_filename,
            mime="application/json"
        )

# Add some instructions or information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app analyzes citations for a given opinion ID. "
    "Enter an opinion ID and click 'Analyze Citations' to see how other cases cite and treat the main case."
    "This is a prototype. Limited to 3 citing cases for now" 
)


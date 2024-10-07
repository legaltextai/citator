import streamlit as st
import requests
import time
import json
import os
import concurrent.futures
import logging
from typing import List, Dict, Any, Tuple, Union, Optional
import google.generativeai as genai 
import typing_extensions as typing
import openai
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title ("Opinion Citation Analyzer") 


BASE_URL = os.getenv('BASE_URL', "https://www.courtlistener.com/api/rest/v4")
AUTH_TOKEN = os.getenv('AUTH_TOKEN', "your_courtlistener_authorization_token")
GENAI_API_KEY = os.getenv('GENAI_API_KEY', "google_gemini_api")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', "your_openai_api_key")

HEADERS = {
    'Authorization': f'Token {AUTH_TOKEN}'
}

genai.configure(api_key=GENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

class Label(str, Enum):
    followed = "followed"
    distinguished = "distinguished"
    partially_overruled = "partially overruled"
    overruled = "overruled"
    rejected = "rejected"
    declined_to_follow = "declined to follow"
    mentioned = "mentioned"

class Color(str, Enum):
    Green = "Green"
    Blue = "Blue"
    Yellow = "Yellow"
    Red = "Red"
    Gray = "Gray"
    Orange = "Orange"
    Purple = "Purple"

class Case(BaseModel):
    name: str
    citation: str

class CitingCase(Case):
    label: Label
    color: Color
    reasoning: str

class CitationAnalysis(BaseModel):
    cited_case: Case
    citing_cases: List[CitingCase] = Field(default_factory=list)

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


with st.expander("Citation Color Legend"):
    st.markdown("""
• ✅ Green (Followed): The citing case adhered to the cited case as a precedent.

• 🔵 Blue (Distinguished): The citing case identified differences between the current case and the cited case, limiting its precedential value.

• ⚠️ Yellow (Partially Overruled): The citing case overturned certain aspects of the cited case while maintaining others.

• ❗ Red (Overruled): The citing case completely overturns the cited case, negating its precedential authority.

• ⚫ Gray (Rejected): The citing case refuses to accept the cited case as a precedent.

• 🔶 Orange (Declined to Follow): The citing case chooses not to adopt the reasoning or decision of the cited case as precedent.

• 🟣 Purple (Mentioned): The citing case references the cited case without adopting or rejecting it as a precedent.
""")

st.markdown("---")

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
        return data.get('results', [])[:20]  # Limit to first opinion
    return []

def process_single_opinion(main_case_name: str, citing_case_name: str, date: str, opinion_text: str) -> Union[dict, None]:
    prompt = f"""Analyze the following opinion text for citations of "{main_case_name}". 
    The citing case name is "{citing_case_name}".

    Use these labels and colors, with examples of how to categorize them:

    1. "followed" (Green): The citing case adhered to the cited case as a precedent.
       Examples: "We follow the decision in Smith v. Nationwide Mut. Ins. Co.", 
                 "For the foregoing reasons, the judgment of the court is AFFIRMED."

    2. "distinguished" (Blue): The citing case identified differences between the current case and the cited case, limiting its precedential value.
       Examples: "The court DISTINGUISHES the holding in Anderson v. Court based on the facts of the present case.",
                 "This Circuit has joined several others in distinguishing the two doctrines."

    3. "partially overruled" (Yellow): The citing case overturned certain aspects of the cited case while maintaining others.
       Example: "The decision in Johnson v. State is PARTIALLY OVERRULED by our ruling."

    4. "overruled" (Red): The citing case completely overturns the cited case, negating its precedential authority.
       Examples: "We overrule the decision in Smith v. Nationwide Mut. Ins. Co.",
                 "The court's refusal to honor the previous judgment effectively REVERSES the earlier decision."

    5. "rejected" (Gray): The citing case refuses to accept the cited case as a precedent.
       Examples: "The court REJECTS the precedent set by Brown v. Board of Education.",
                 "The decision of the trial court is REV'D."

    6. "declined to follow" (Orange): The citing case chooses not to adopt the reasoning or decision of the cited case as precedent.
       Examples: "The Supreme Court DECLINES TO FOLLOW the lower court's interpretation of the statute.",
                 "This Circuit declines to follow the reasoning established in prior cases."

    7. "mentioned" (Purple): The citing case references the cited case without adopting or rejecting it as a precedent.
       Examples: "The opinion refers to Johnson v. State.",
                 "Several courts have mentioned the ruling in Miller v. State without fully endorsing it."

    Choose the most appropriate label based on the opinion's language and context. 
    Provide a very detailed and long reasoning, in bullet points, for each classification. Add quotes from the opinion text that support your choice.

    Opinion Text:
    {opinion_text[:400000]} #limited to 400,000 characters, which is approx 80K words, which is approx 120K tokens, the limit for gpt-4o
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        temperature=0,
        max_tokens=10000,
        messages=[
            {
                "role": "system",
                "content": "You are a legal analyst tasked with extracting citation information from legal opinions.",
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        tools=[
            openai.pydantic_function_tool(CitationAnalysis),
        ],
    )

    try:
        result = completion.choices[0].message.tool_calls[0].function.parsed_arguments
        if isinstance(result, CitationAnalysis):
            result = result.model_dump()
        if not isinstance(result, dict):
            result = {}
        if 'citing_cases' not in result:
            result['citing_cases'] = []
        return result
    except Exception as e:
        print(f"Error processing opinion {citing_case_name}: {str(e)}")
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

if "opinion_id" not in st.session_state:
    st.session_state.opinion_id = ""

def set_search_input(value):
    st.session_state.opinion_id = value

def on_text_input_change():
    st.session_state.opinion_id = st.session_state.temp_opinion_id

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


st.write("Try one of these cases or enter opinion_id")
st.button("Plessy v Ferguson", on_click=set_search_input, args=("94508",))
st.button("Marbury v Madison", on_click=set_search_input, args=("84759",))


st.text_input("Enter Opinion ID and press Enter:", key="temp_opinion_id", on_change=on_text_input_change)

if st.session_state.opinion_id:
    with st.spinner("Fetching main case name..."):
        main_case_name = get_case_name(st.session_state.opinion_id)
        st.write(f"Main Case: {main_case_name}")

    with st.spinner("Processing citing opinions..."):
        main_case_name, results = process_opinion(st.session_state.opinion_id)


    COLOR_ICONS = {
        "Green": ("✅", "green"),
        "Blue": ("🔵", "blue"),
        "Yellow": ("⚠️", "yellow"),
        "Red": ("❗", "red"),
        "Gray": ("⚫", "gray"),
        "Orange": ("🔶", "orange"),
        "Purple": ("🟣", "purple")
    }

    if results:
        st.success(f"Successfully processed {len(results)} citing opinions for {main_case_name}")
        
        for result in results:
            citing_cases = result.get('citing_cases', [])
            if not citing_cases:
                st.warning("No citing cases found for this result")
                continue

            for citing_case in citing_cases:
                # Get color and icon
                color = citing_case.get('color', 'Unknown')
                if isinstance(color, Enum):
                    color = color.value
                icon, text_color = COLOR_ICONS.get(color, ("❓", "black"))

                # Create expander title with icon
                expander_title = f"{icon} {citing_case.get('name', 'Unknown Case')}"
                
                with st.expander(expander_title):
                    cited_case = result.get('cited_case', {})
                    st.markdown(f"**Cited Case:** {cited_case.get('name', 'Unknown')}")
                    st.markdown(f"**Cited Case Citation:** {cited_case.get('citation', 'Unknown')}")
                    
                    st.markdown(f"**Citing Case Citation:** {citing_case.get('citation', 'Unknown')}")
                    
                    # Updated treatment display with visual cues
                    label = citing_case.get('label', 'Unknown')
                    
                    # Extract the value from the Enum if it's an Enum
                    if isinstance(label, Enum):
                        label = label.value
                    
                    st.markdown(f"**Label:** <span style='color: {text_color};'>{label.capitalize()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Color:** {color}")
                    
                    st.markdown("**Reasoning:**")
                    st.markdown(citing_case.get('reasoning', 'No reasoning provided'))
                    
                    # Display all other available information
                    for key, value in citing_case.items():
                        if key not in ['name', 'citation', 'label', 'color', 'reasoning']:
                            st.markdown(f"**{key.capitalize()}:** {value}")
    else:
        st.warning(f"No results found for {main_case_name}")

    output_filename = f'processed_opinions_{st.session_state.opinion_id}.json'
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

st.sidebar.header("About")
st.sidebar.info(
    """
    This app analyzes citations for a given opinion ID.
    
    Enter an opinion ID and click 'Analyze Citations' to see how other cases cite and treat the main case.
    
    Limited to 20  citing cases for now. Limited to approx 80,000 first words in the opinion text for the prompt. 
    
    This is a prototype. 
    """
)

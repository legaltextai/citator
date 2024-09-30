# Citator Framework

## Purpose
This project is my attempt to build a framework for assessing whether a case is "good law" by analyzing how it's cited and treated in subsequent cases.

## Key Components
- Court Listener API integration for retrieving case data
- Google's Generative AI for analyzing case citations and treatments
- Parallel processing of citing opinions
- JSON output of citation analysis results

## Usage


## Performance Variables
- Rate limiting and API response times (ensure you have obtained Courtlistener authorization token)
- Quality and completeness of case text data
- Accuracy of the language model in classifying citations
- Number of citing opinions processed

## Classification Matrix
The current classification system uses the following labels:
- `followed`
- `distinguished`
- `partially overruled`
- `overruled`
- `rejected`
- `mentioned`

Further development should focus on:
1. Refining classification criteria
2. Implementing a confusion matrix for model evaluation
3. Collecting human-labeled data for validation
4. Evaluating and synthesizing classification approaches from Lexis, Westlaw, Paxton, Casetext, etc. 

## Next Steps
1. Implement error handling and data validation
2. Develop a more robust citation extraction method
3. Integrate additional legal databases for comprehensive analysis
4. Create a web interface to use as a playground. 
5. Implement unit tests and integration tests
6. Switching from API requests to direct calls to CL's Postgres database.
7. Evaluating the completeness of _'opinions-cited' API_[https://www.courtlistener.com/api/rest/v4/opinions-cited/ ](url) vs CL citation map vs search API for the purposes of extracting all citing cases. 


## Limitations
- Limited to CourtListener API data
- Depends on the accuracy of the language model
- Currently processes a small sample of citing opinions


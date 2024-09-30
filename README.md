# Court Decision Status Analysis Framework

## Purpose
This project is an initial attempt to build a framework for assessing whether a case is "good law" by analyzing how it's cited and treated in subsequent cases.

## Key Components
- Court Listener API integration for retrieving case data
- Google's Gemini Pro for analyzing case citations and treatments (due to its nearly unlimited context size) 
- Parallel processing of citing opinions
- JSON output of citation analysis results

## Usage
1. Set environment variables:
   - `AUTH_TOKEN`: Court Listener API authentication token
   - `GENAI_API_KEY`: Google Generative AI API key

2. There are two options to run: as a streamlit app, or a python script 

3. Results are saved in `processed_opinions_{opinion_id}.json` 

4. As this is a prototype, the results are limited to 3 citing opinions for now. 

## Performance Variables
- Rate limiting and API response times
- Quality and completeness of case text data
- Accuracy of the language model in classifying citations
- Number of citing opinions processed

## Classification Matrix
The current classification system uses the following labels:
- followed
- distinguished
- partially overruled
- overruled
- rejected
- mentioned

Further development should focus on:
1. Refining classification criteria
2. Implementing a confusion matrix for model evaluation
3. Collecting human-labeled data for validation

## Next Steps
1. Implement error handling and data validation
2. Develop a more robust citation extraction method. 
3. Assess the completeness of API search ("cited") vs API opinions-cited vs citation map in bulk files.
4. Learn from and synthesize classification matrix based on Lexis, Westlaw, Casetext, Paxton labeling 
5. Create a web interface for easier interaction with the tool. Use as a playground to tweak variables. 

## Limitations
- Limited to Court Listener API data
- Depends on the accuracy of the language model
- Currently processes a small sample of citing opinions

## Contributing
Contributions are welcome. Please open an issue to discuss proposed changes or improvements.


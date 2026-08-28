import pdfplumber
from pathlib import Path
import json
from llm.groq_client import chat
from llm.prompts import SYSTEM_PROMPT

def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def strip_thinking(text: str) -> str:
    """Strip reasoning/thought block tags like <think>...</think> from text."""
    if "<think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            return text.split("<think>")[0].strip()
    return text

def extract_metadata_via_llm(raw_text: str) -> dict:
    """Extract structured metadata from judgment text using an LLM."""
    truncated = raw_text[:6000] # Fit in standard context window (avoids 413 Entity Too Large)
    
    prompt = f"""
You are a legal metadata extraction AI. Extract the following fields from the provided judgment text and return them as a valid JSON object. Do not return any other text, only the JSON.

Keys required:
- "case_no": (string or null) The official case number.
- "date": (string or null) The date of the judgment (e.g. '19 May, 1950').
- "petitioner": (string or null) The name of the petitioner / appellant.
- "respondent": (string or null) The name of the respondent / defendant.
- "judges": (string or null) The name(s) of the judges.
- "subject": (string or null) A brief 1-5 word category or subject of the case.
- "acts": (string or null) Any specific acts or statutes mentioned.
- "verdict": (string or null) A short summary of the outcome / verdict.

Judgment Text:
{truncated}
"""
    # Using a fast available model for metadata extraction
    resp = chat("groq/compound-mini", "You are a helpful JSON extraction assistant.", prompt, max_tokens=1024, temperature=0.1)
    
    if resp["error"] or not resp["text"]:
        # Fallback to qwen if compound-mini fails
        resp = chat("qwen/qwen3.6-27b", "You are a helpful JSON extraction assistant.", prompt, max_tokens=1024, temperature=0.1)
        if resp["error"] or not resp["text"]:
            return {}
        
    try:
        # Clean think tags first
        result_text = strip_thinking(resp["text"]).strip()
        
        # Find json block if any markdown formatting was used
        if result_text.startswith("```json"):
            result_text = result_text.split("```json")[1]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
        data = json.loads(result_text.strip())
        
        # Normalize keys in case LLM used synonyms or different case styles
        normalized = {}
        key_mappings = {
            "case_no": ["case_no", "case_number", "caseno"],
            "date": ["date", "date_of_judgment", "judgment_date", "decision_date"],
            "petitioner": ["petitioner", "appellant", "plaintiff"],
            "respondent": ["respondent", "defendant"],
            "judges": ["judges", "judge", "bench", "coram"],
            "subject": ["subject", "category", "topic", "headnote", "head_note"],
            "acts": ["acts", "act", "acts_cited", "statutes"],
            "verdict": ["verdict", "decision", "outcome", "holding"],
        }
        
        for target_key, aliases in key_mappings.items():
            val = None
            for alias in aliases:
                if alias in data:
                    val = data[alias]
                    break
            if val is None:
                # Case-insensitive / character-flexible backup match
                for k, v in data.items():
                    clean_k = k.lower().replace("_", "").replace(" ", "")
                    if clean_k == target_key.lower():
                        val = v
                        break
            
            # If the value is a list (e.g. for bench/judges/acts), join it as a string
            if isinstance(val, list):
                val = ", ".join(str(item) for item in val)
            
            normalized[target_key] = val
            
        return normalized
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON from LLM: {e}\nRaw output: {resp['text']}")
        return {}

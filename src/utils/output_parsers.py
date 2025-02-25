import re
from typing import List

def parse_statements_llama(text):
    statements = []
    # Strip initial "assistant\n\n" if present
    text = text.replace("SQL query(s):", "")
    text = text.strip()
    if text.lower().startswith("assistant\n\n"):
        text = text[len("assistant\n\n"):].strip()
    
    # Extract code blocks within triple backticks
    pattern = r"```(?:\w+)?\s+(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        statements = matches  # Directly use the matches list
    else:
        statements = [text]  # Use the whole input if no code blocks are found

    new_statements = []
    for code in statements:
        if "\n\n" in code or ";" in code:
            # Split on newlines or semicolons
            split = re.split(r';|\n\n', code)
            new_statements += [
                x.strip() for x in split 
                if x.strip() and (x.strip().lower().startswith("select") or x.strip().lower().startswith("with"))
            ]
    if new_statements:
        statements = new_statements
    
    if not statements and (text.strip().lower().startswith("select") or text.strip().lower().startswith("with")):
        statements = [text.strip()]

    if not statements:
        print(f'Something wrong: {text}')
        statements = []
    
    # Final cleanup
    new_statements = [stat.replace("\\", "") for stat in statements]
    statements = new_statements

    return statements

def parse_ambig_detection(text):
    text = text.strip()
    if text.lower().startswith("assistant\n\n"):
        text = text[len("assistant\n\n"):].strip()

    if "yes" in text.lower():
        return "yes"
    elif "no" in text.lower():
        return "no"
    else:
        return 'n/a'
    
def parse_interpretations(text: str) -> List[str]:
    """Parse model output into list of interpretations"""
    # Remove any system/assistant prefixes and any text before the actual interpretations
    text = re.sub(r'^(?:system|assistant):\s*(?:Here are the interpretations:)?\s*', '', text, flags=re.IGNORECASE)
    
    # First try to extract code blocks
    code_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
    if code_blocks:
        # For each code block, split into lines and clean numbered lists
        interpretations = []
        for block in code_blocks:
            if not block.strip():
                continue
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            for line in lines:
                # Remove leading numbers or bullet points
                cleaned_line = re.sub(r'^\s*(?:\d+\.|[-•])\s*', '', line)
                interpretations.append(cleaned_line)
        return interpretations
    
    # Split the text into lines first
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # If the first line doesn't start with a number or bullet, keep it as is
    # For subsequent lines, remove any leading numbers or bullets
    interpretations = []
    for line in lines:
        # Remove leading numbers or bullet points
        cleaned_line = re.sub(r'^\s*(?:\d+\.|[-•])\s*', '', line)
        interpretations.append(cleaned_line)
    
    return interpretations

"""
Programming Experiment 
"""

import os
import json
import random
import time
import re  
import pandas as pd  
from datetime import datetime
from models import LLMModel
from config import MODELS, RESULTS_DIR, API_KEY, API_BASE, STREAMING, DEFAULT_TEMPERATURE

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def init_llm():
    """Initialize all large language models"""
    llm_models = []
    for model_config in MODELS:
        model = LLMModel(
            name=model_config["name"],
            description=model_config["description"],
            api_key=model_config["api_key"],
            streaming=model_config.get("streaming") # Get model-specific streaming setting
        )
        llm_models.append(model)
    return llm_models

def format_programming_prompt_question():
    """Build prompt for programming question"""
    return """You are a senior programming competition question setter. Your task is to perform the following steps strictly and output only the final result in JSON format with no Markdown, no commentary, and no additional fields or nesting.
1.Study how difficult programming problems are framed in high-level competitive programming. Identify what makes them difficult: novel domains, tight constraints and implementation complexity.
(1)Hotel in Ves Lagos
Time limit per test: 0.25 second(s)
Memory limit: 262144 kilobytes
input: standard
output: standard
A new hotel is being built in the city of Ves Lagos. The hotel will have an infinite number of rooms (it is out of fashion to build hotels with finite numbers of rooms). The new hotel also tries to cater for superstitious guests.
The most common superstition in Ves Lagos is that the number 13 brings bad luck. Accordingly, only numbers whose decimal forms do not contain the substring "13" will be used to label the rooms in the new hotel. For example, the hotel will have rooms numbered 1, 3, 14, 31, 123, but will not have the rooms 13, 132, 913, 1308, 1313.
Let's consider the list of all room numbers, ordered increasingly. Find the N-th number in this list (members of the list are indexed from 1).
Input
The input file contains several test cases. The first line of the file contains T (1 ≤ T ≤ 100), the number of test cases. Each of the following T lines describes one test case and contains the integer N (1 ≤ N ≤ 1018).
Output
The output file should contain exactly T lines, with the i-th line containing exactly one integer, the answer for the i-th test case from the input file.
Example(s)
sample input
3
20
150
1
sample output
21
162
1

(2)"North-East"
Time limit per test: 0.5 second(s)
Memory limit: 262144 kilobytes
input: standard
output: standard
The popular music band of international fame "North-East" is coming to Berland! This news has spread all over the country, so numerous fans are now ready to rush and buy all the tickets!
At present the fans still don't know in which cities the band plans to give concerts. The only thing is known at the moment is that the band will visit several cities, and as their name says, they will strictly move north and east when going to the next city. In other words when the band moves from city i to city j, city j is always located northward and eastward of the city i.
It's also known that the tour is planned in such a way that the maximum possible number of cities will be visited. The musicians refuse to reveal other details. As you know, fans always get ready for the arrival of their idols, so they would appreciate any single detail about possible movements of their favorite musicians.
Your task is to help the fans and find two lists of cities — A and B. The first list A should contain the cities, which the band might visit during the tour. The second list B should contain the cities, which the band will have to visit for sure during the tour.
Input
The first line of input contains a single integer n (1 ≤ n ≤ 105) — amount of cities in the country. The following n lines contain coordinates of the cities. Each line contains a pair of integers xi, yi (-106 ≤ xi, yi ≤ 106) — the coordinates of the i-th city. Ox axis is directed west-to-east, and Oy axis — south-to-north. No two given cities will be located at the same point.
Output
Print the required list A to the first line of output and B to the second line. Each list should start with the amount of cities in it, followed by the indices of cities in increasing order. Cities are numbered from 1 to n.
Example(s)
sample input
5
3 2
1 1
5 5
2 3
4 4
sample output
5 1 2 3 4 5
3 2 3 5
(3)"Droid formation"
Time limit per test: 0.5 second(s)
Memory limit: 262144 kilobytes
input: standard
output: standard
A long time ago (but somehow in the future), in a Galaxy Far Far Away...
— Your majesty, Jedi detachment almost finished to mine our new Death Cube! New battle droids are unable to restrain them! What to do!?
— Rest assured! What is the strength of every troop of droids?
— 12 droids, your majesty!
— Fools! I told you that a troop should have 4 variants of evolution but troop of 12 droids has 6! This perverts threads of the Power — and infeeds Jedis! Regroup the army — and Jedis will lose!
— Yes sir!
Number of variants of 	evolution of a troop of droids is the number of ways to draw it up in rows so that every row has the same number of droids. For example, a troop of 12 droids can be arranged in 1 row of 12 droids, 2 rows of 6 droids, 3 rows of 4 droids, 4 rows of 3 droids, 6 rows of 2 droids and 12 rows consisting of 1 droid each. So, as the Emperor noticed, there are 6 variants of evolution for this troop of droids.
You problem is more general — given the number K of favorable variants of evolution, find the smallest positive size of a troop of droids N which has this very number of variants of evolution.
Input
Input file contains only number K from the problem statement (1 ≤ K ≤ 105).
Output
Write to the output file the required number N. If there is no such number, write to the output file number 0 instead.
Example(s)
sample input
sample output
4
6
2.Create a completely original and non-derivative programming problem that includes:
A concise but fully specified problem description.
Input and output format.
Clear constraints.
3.Ensure the problem reflects true conceptual depth and comes from an original domain, not just a variant of classic problems (e.g., not just another sorting or graph problem with tweaks).
4.Solve the problem fully, and include:
A complete working implementation of the solution (in plain text).
Explanations of the core algorithm and ideas used.
Time and space complexity analysis.
5.Define a scoring criteria out of 100 points. Evaluate from the following but not limited to several aspects:
Accuracy and correctness
Execution speed (runtime performance)
Memory efficiency
Human readability of code
Code modularity and organization
6.VERY IMPORTANT:
Do NOT output any extra text, explanations, section headers, or Markdown formatting.
All output must be in JSON format only.
Use EXACTLY the following three fields:
json
{
  "Problem": "Your newly created difficult programming problem with clear requirements and constraints.",
  "Standard Solution": "Complete solution including FULL CODE IMPLEMENTATION with key concept explanations and complexity analysis.",
  "Scoring Standard": "Detailed scoring criteria across multiple evaluation dimensions, out of 100."
}
7.Do NOT change the field names ("Problem", "Standard Solution", "Scoring Standard"). Do not include any other fields. Output must be a flat JSON object with only those three keys.
"""

def format_programming_prompt_answer(problem):
    """Build prompt for answering programming problems"""
    return f"""Please solve the following programming problem with a complete code implementation only.
Problem: {problem}
Requirements:
Solve the problem independently using the programming language you consider most appropriate.
Your code must be complete and runnable (not pseudocode or algorithm outline).
Include all necessary functions, classes, and logic.
Handle edge cases appropriately.
Follow good programming practices with clear variable names and inline comments if needed.
IMPORTANT: 
DO NOT include any explanations, thoughts, steps, or discussions.
DO NOT output anything except a JSON object with one key: "Answer"
The "Answer" field must contain ONLY the actual code, properly escaped as a single-line JSON string (with \\n for newlines, and \\\" for quotes if needed).
DO NOT wrap the code in Markdown, and DO NOT include any formatting symbols like backticks or section headers.
Any reasoning, description, or justification MUST be omitted completely from the final output.
Final Output Format (Strictly required):
{{
  "Answer": "your_full_code_here_with_escaped_newlines"
}}
Notes:
This is a strict format enforcement. Any deviation (extra fields, formatting, explanations, etc.) will be considered incorrect.
The response must be valid JSON, flat, and contain only the "Answer" key with actual executable code as its value.
"""

def format_programming_prompt_evaluate(problem, standard_solution, scoring_standard, all_answers, evaluator_name):
    """Build prompt for evaluating answers"""
    # Create all answers except the evaluator's own answer
    if evaluator_name is None:
        # Questioner evaluates all answers
        filtered_answers_str = json.dumps(all_answers, ensure_ascii=False, indent=2)
    else:
        # Responder evaluates answers except their own
        filtered_answers = {k: v for k, v in all_answers.items() if k != evaluator_name}
        filtered_answers_str = json.dumps(filtered_answers, ensure_ascii=False, indent=2)
    
    return f"""Evaluate the responses from multiple large language models based on the following information:
Problem:
{problem}

Standard Solution:
{standard_solution}

Scoring Standard:
{scoring_standard}

All the Models' Code Implementations (excluding your own):
{filtered_answers_str}
Please perform the following tasks:
1.Evaluate each model's answer based on the standard_solution and scoring_standard.
2.For each model, provide a precise score out of 100 based on the scoring standard. Models that didn't provide actual code implementation should receive significantly lower scores.
3.Provide detailed reasoning for each score, highlighting both strengths and weaknesses in the implementation.
4.If any model's solution exceeds the standard solution in quality, efficiency or elegance, note this explicitly.
5.Output your evaluation results in JSON format:
{{
  "Scores": {{
    "Model1": 95,
    "Model2": 80,
    "Model3": 70
  }},
  "Reason": {{
    "Model1": "Detailed explanation of scoring, including specific assessment of code implementation.",
    "Model2": "Detailed explanation...",
    "Model3": "Detailed explanation..."
  }}
}}
"""

def extract_json_response(response):
    """Extract JSON data from response"""
    if not response:
        print("API return empty response")
        return None
        
    # Try to parse directly
    try:
        json_data = json.loads(response)
        
        # If parsing is successful, handle some field mappings
        # For example, some models might use 'Solution' instead of 'Standard Solution'
        if 'Standard Solution' not in json_data and 'Solution' in json_data:
            solution = json_data['Solution']
            if isinstance(solution, dict) and 'Code Implementation' in solution:
                # Some models might put code under Solution.Code Implementation
                json_data['Standard Solution'] = solution['Code Implementation']
            elif isinstance(solution, str):
                # Some models might directly provide a solution in the Solution field
                json_data['Standard Solution'] = solution
        
        return json_data
    except json.JSONDecodeError:
        pass
    
    # Process LaTeX formulas and special characters in the response
    cleaned_response = response
    try:
        # Escape LaTeX formulas
        latex_patterns = [
            r'\\\[.*?\\\]',  # \[ ... \]
            r'\\\(.*?\\\)',  # \( ... \)
            r'\$\$.*?\$\$',  # $$ ... $$
            r'\$.*?\$'       # $ ... $
        ]
        for pattern in latex_patterns:
            cleaned_response = re.sub(pattern, lambda m: m.group(0).replace('\\', '\\\\'), cleaned_response, flags=re.DOTALL)
    except Exception:
        # If processing fails, continue with the original response
        cleaned_response = response
    
    # First try to find JSON in the processed response
    try:
        # Find the outermost curly braces
        start = cleaned_response.find('{')
        end = cleaned_response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = cleaned_response[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, try to fix common JSON formatting issues
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                # Try to fix unclosed quotes
                json_str = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', json_str)
                # Try parsing again
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # Try to use the original response for further processing, looking for answers with explicit markers
    try:
        # Find the outermost curly braces
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, check if it's a formatting issue
                # Remove all newlines and extra spaces
                json_str = re.sub(r'\s+', ' ', json_str).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # Look for the ending JSON block - usually ends with "Final Answer:" or similar text
    try:
        # Find the ending JSON block
        final_answer_patterns = [
            r'(?:Final Answer:|最终答案:|最后答案:|Answer:|结果:)\s*(\{.*\})',
            r'(?:The final result is:)\s*(\{.*\})',
            r'(?:My final answer is:)\s*(\{.*\})'
        ]
        
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        # Try to fix formatting
                        cleaned_match = re.sub(r'\s+', ' ', match).strip()
                        try:
                            return json.loads(cleaned_match)
                        except:
                            continue
    except Exception:
        pass
    
    # Try to find JSON object using regex
    try:
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.finditer(json_pattern, response)
        for match in matches:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                # Verify if it contains expected fields
                if any(key in result for key in ["Answer", "Scores", "Problem", "Standard Solution", "Scoring Standard"]):
                    return result
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # Manual extraction of key information
    try:
        # Try to extract Scores and Reason
        scores = {}
        reasons = {}
        
        # Find Scores section
        if "Scores" in response:
            scores_section = response[response.find("Scores"):]
            model_score_pattern = r'"([^"]+)"\s*:\s*(\d+(?:\.\d+)?)'
            for match in re.finditer(model_score_pattern, scores_section):
                model_name = match.group(1)
                score = float(match.group(2))
                scores[model_name] = score
        
        # Find Reason section
        if "Reason" in response:
            reason_section = response[response.find("Reason"):]
            model_pattern = r'"([^"]+)":\s*"([^"]+)"'
            for match in re.finditer(model_pattern, reason_section):
                model_name = match.group(1)
                reason = match.group(2)
                reasons[model_name] = reason
        
        if scores or reasons:
            result = {}
            if scores:
                result["Scores"] = scores
            if reasons:
                result["Reason"] = reasons
            return result
    except Exception:
        pass
    
    # Check for "Answer" keyword
    try:
        # More flexible detection of Answer format
        patterns = [
            r'"Answer"\s*:\s*"([^"]+)"',  # Standard JSON format
            r'"Answer"\s*:\s*([^",\}\s][^",\}]*)',  # Value without quotes
            r'Answer:\s*([^\n]+)',  # Key-value pair but not in JSON
            r'答案:\s*([^\n]+)',  # Chinese mode
            r'answer is\s*:?\s*([^\n,.]+)',  # Natural language expression
            r'(?:conclusion|总结)[\s:]*([^\n]{10,})'  # Conclusion part might contain answer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                return {"Answer": answer_text}
    except Exception:
        pass
    
    # Try to extract a dictionary containing problem, answer, and scoring standard
    try:
        # Check if required keywords are present
        if "Problem" in response and "Standard Solution" in response and "Scoring Standard" in response:
            # Extract content to create a manually built JSON
            problem_start = response.find("Problem") + len("Problem") + 1
            problem_end = response.find("Standard Solution") - 1
            problem = response[problem_start:problem_end].strip().strip('":,')
            
            answer_start = response.find("Standard Solution") + len("Standard Solution") + 1
            answer_end = response.find("Scoring Standard") - 1
            standard_solution = response[answer_start:answer_end].strip().strip('":,')
            
            scoring_start = response.find("Scoring Standard") + len("Scoring Standard") + 1
            scoring_standard = response[scoring_start:].strip().strip('":,')
            
            return {
                "Problem": problem,
                "Standard Solution": standard_solution,
                "Scoring Standard": scoring_standard
            }
    except Exception:
        pass
    
    # If it's an answer case, try to extract the last few lines as the answer
    if "answer" in response.lower() or "答案" in response or "conclusion" in response.lower() or "总结" in response:
        try:
            # Get the last few lines
            lines = response.split('\n')
            # Find lines containing JSON
            json_lines = [line for line in lines if '{' in line and '}' in line]
            if json_lines:
                for line in reversed(json_lines):  # Start searching from the end
                    try:
                        json_start = line.find('{')
                        json_end = line.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = line[json_start:json_end]
                            return json.loads(json_str)
                    except:
                        continue
            
            # If no JSON line found, try to extract the last few lines as the answer
            last_lines = [line for line in lines[-7:] if len(line.strip()) > 10]  # Ensure line content is sufficient
            if last_lines:
                return {"Answer": last_lines[-1].strip()}
        except Exception:
            pass
        
    # Handle code blocks as answers 
    try:
        code_blocks = []
        # Try to find code blocks
        code_patterns = [
            r'```(?:python)?\n(.*?)\n```',  # Standard Markdown code block
            r'```(?:python)?(.*?)```',       # Code block without newlines
            r'<code>(.*?)</code>',           # HTML code tag
            r'def\s+[a-zA-Z0-9_]+\s*\(.*?\):(.*?)(?:def\s|class\s|if\s+__name__|\Z)', # Python function
            r'class\s+[a-zA-Z0-9_]+.*?:(.*?)(?:def\s|class\s|if\s+__name__|\Z)',      # Python class
            r'import.*?\n.*?if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:(.*?)(?:\Z)'   # Full Python program
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code_blocks.extend(matches)
        
        if code_blocks:
            # Select the longest code block as the answer
            longest_code = max(code_blocks, key=len)
            if len(longest_code.strip()) > 20:  # Ensure code block has a certain length
                return {"Answer": longest_code.strip()}
    except Exception:
        pass
    
    # Locate and extract possible code implementation parts
    try:
        # Find common code prompts
        code_indicators = [
            "Here's my solution:", "Here's the code:", "Implementation:", 
            "My code implementation:", "Here's my implementation:", 
            "Complete code:", "代码实现:", "代码如下:", "以下是代码:"
        ]
        
        for indicator in code_indicators:
            if indicator in response:
                start_idx = response.find(indicator) + len(indicator)
                # Try to find the end position of the code section
                end_markers = ["Explanation:", "Time complexity:", "Space complexity:", 
                               "Note:", "解释:", "时间复杂度:", "空间复杂度:", "\n\n\n"]
                end_positions = [response.find(marker, start_idx) for marker in end_markers 
                                if response.find(marker, start_idx) > -1]
                
                if end_positions:
                    end_idx = min(end_positions)
                    code_section = response[start_idx:end_idx].strip()
                else:
                    # If no end marker found, take the remaining text
                    code_section = response[start_idx:].strip()
                
                if len(code_section) > 50 or 'def ' in code_section or 'class ' in code_section:
                    return {"Answer": code_section}
    except Exception:
        pass
    
    # Print part of the response for debugging
    print(f"Unable to extract JSON from response, here is the original response")
    print("-" * 50)
    print(response[:500])
    print("-" * 50)
    
    # Final attempt - return original response as Answer
    max_length = 1000
    answer_text = response.strip()
    if len(answer_text) > max_length:
        answer_text = answer_text[:max_length] + "..."
    return {"Answer": answer_text}

def print_separator():
    """Print separator line"""
    print("\n" + "="*80 + "\n")

def normalize_model_name(name):
    """Normalize model name, handle case inconsistency issues"""
    # Check if it's an invalid model name
    invalid_names = ["standard answer", "standard", "answer", "model", "none", "null", "undefined", "unknown"]
    if not name or name.lower() in invalid_names:
        return None  # Return None if it's not a valid model name
        
    # Convert to lowercase for matching
    name_lower = name.lower()
    
    # Get all model names and descriptions from config
    known_models = {}
    for model_config in MODELS:
        model_name = model_config["name"]
        description = model_config["description"]
        known_models[model_name.lower()] = (model_name, description)
    
    # Standard model name mapping (lowercase -> standard name)
    name_mapping = {}
    
    # First add exact matches
    for model_name, (std_name, desc) in known_models.items():
        name_mapping[model_name] = std_name
    
    # Then add aliases and abbreviations, but avoid confusion with generic names
    for model_name, (std_name, desc) in known_models.items():
        # GPT series models
        if "gpt-4.1" in model_name:
            name_mapping["gpt4.1"] = std_name
            name_mapping["gpt4"] = std_name
            name_mapping["gpt-4.1"] = std_name

        # Claude series models
        elif "claude-3-7-sonnet" in model_name:
            name_mapping["claude-3-7"] = std_name
            name_mapping["claude-3.7"] = std_name
            name_mapping["claude3.7"] = std_name
            name_mapping["claude-sonnet"] = std_name
            
        # Deepseek series models need precise distinction
        elif "deepseek-v3" == model_name:
            name_mapping["deepseek-v3"] = std_name
            name_mapping["deepseekv3"] = std_name
            name_mapping["deepseek-coder"] = std_name  # Common alias
            
        elif "deepseek-r1" == model_name:
            name_mapping["deepseek-r1"] = std_name
            name_mapping["deepseekr1"] = std_name
            
        # Qwen series models
        elif "qwen2.5-max" == model_name:
            name_mapping["qwen2.5"] = std_name
            name_mapping["qwen-max"] = std_name
            name_mapping["qwen-2.5"] = std_name
                  
        # O3-mini model
        elif "o3-mini" in model_name:
            name_mapping["o3-mini"] = std_name
            name_mapping["o3mini"] = std_name
            name_mapping["o3"] = std_name
            
        # O1 model
        elif "o1" == model_name:
            name_mapping["o1"] = std_name
            
        # Gemini model
        elif "gemini" in model_name:
            name_mapping["gemini-pro"] = std_name
            name_mapping["gemini-2.5"] = std_name
            name_mapping["gemini2.5"] = std_name
            name_mapping["gemini"] = std_name

    # Try exact matching
    if name_lower in name_mapping:
        return name_mapping[name_lower]
    
    # Try matching with name and description
    for model_name, (std_name, desc) in known_models.items():
        # Check if a match is found in name or description
        if name_lower in model_name or (desc and name_lower in desc.lower()):
            return std_name
    
    # Try partial matching, but be more strict
    # First try to find a match containing a specific version number
    for key, value in name_mapping.items():
        if (key in name_lower) and any(version_marker in key for version_marker in ["v3", "r1", "4o", "3-7", "2.5", "4v", "mini"]):
            return value
    
    # Other common abbreviations, but prevent confusion
    if "deepseek" in name_lower:
        # If version info is included, try precise matching
        if "v3" in name_lower or "coder" in name_lower:
            for model_name, (std_name, _) in known_models.items():
                if "deepseek-v3" == model_name:
                    return std_name
        elif "r1" in name_lower:
            for model_name, (std_name, _) in known_models.items():
                if "deepseek-r1" == model_name:
                    return std_name
        # If no clear version, do not infer, return original name
        return name
    
    # Handle other model abbreviations
    basic_mappings = {
        "gpt": "gpt-4.1",
        "claude": "claude-3-7-sonnet-20250219",
        "qwen": "qwen2.5-max",
        "gemini": "gemini-2.5-pro-exp-03-25",
        "o3": "o3-mini",
        "o1": "o1"
    }
    
    for key, value in basic_mappings.items():
        if key in name_lower:
            # Verify if the model exists in MODELS
            for model_name, (std_name, _) in known_models.items():
                if value.lower() in model_name.lower():
                    return std_name
    
    # If no match, return original name
    return name

def get_current_ranking(total_scores):
    """Calculate current ranking based on total scores"""
    # Sort by score in descending order
    sorted_models = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Handle cases where scores are the same
    ranking = []
    current_rank = 1
    current_score = None
    tied_count = 0
    
    for i, (model_name, score) in enumerate(sorted_models):
        if score != current_score:
            current_rank = i + 1
            current_score = score
            tied_count = 0
        else:
            tied_count += 1
            
        ranking.append((current_rank, model_name, score, tied_count > 0))
    
    return ranking

def display_total_ranking(ranking):
    """Display the total ranking"""
    print("\n[Current Total Ranking]")
    print("-" * 80)
    print("Rank   Model name                            Total Score")
    print("-" * 80)
    
    for rank, model_name, score, is_tied in ranking:
        # Only take model name, not description
        short_name = model_name.split()[0] if ' ' in model_name else model_name
        rank_display = f"{rank}{'*' if is_tied else ' '}"  # Tie ranking marker
        print(f"{rank_display:<5}{short_name:<35}{score:.2f}")
        
    print("-" * 80)
    print("Note: * indicates tied ranking")

def normalize_question_data(data):
    """Normalize question data, ensure consistent formatting of fields"""
    if not data:
        return {
            "Problem": "",
            "Standard Solution": "",
            "Scoring Standard": ""
        }
    
    normalized = {}
    
    # Extract problem
    if "Problem" in data:
        normalized["Problem"] = data["Problem"]
    elif "Description" in data:
        normalized["Problem"] = data["Description"]
    else:
        normalized["Problem"] = ""
    
    # Extract standard solution
    if "Standard Solution" in data:
        normalized["Standard Solution"] = data["Standard Solution"]
    elif "Solution" in data:
        solution = data["Solution"]
        if isinstance(solution, dict) and "Code Implementation" in solution:
            normalized["Standard Solution"] = solution["Code Implementation"]
        elif isinstance(solution, str):
            normalized["Standard Solution"] = solution
        else:
            normalized["Standard Solution"] = str(solution)
    else:
        normalized["Standard Solution"] = ""
    
    # Extract scoring standard
    if "Scoring Standard" in data:
        scoring = data["Scoring Standard"]
        if isinstance(scoring, dict):
            # Convert dictionary to formatted string
            scoring_str = "Scoring Standard:\n"
            for criterion, points in scoring.items():
                scoring_str += f"- {criterion}: {points} points\n"
            normalized["Scoring Standard"] = scoring_str
        else:
            normalized["Scoring Standard"] = str(scoring)
    else:
        normalized["Scoring Standard"] = ""
    
    return normalized

def run_programming_experiment():
    """Run programming experiment"""
    # Initialize models
    models = init_llm()
    print(f"Successfully initialized {len(models)} models")
    
    # Create experiment results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"programming_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize total score tracking
    total_scores_by_round = {model.name: [] for model in models}  # Store scores per round
    all_round_scores = []
    
    # Each model asks a question
    for round_num, questioner in enumerate(models, 1):
        print_separator()
        print(f"======= Start of the {round_num} round experiment =======")
        # Only display model name, not description
        questioner_name = questioner.name.split()[0] if ' ' in questioner.name else questioner.name
        print(f"\nQuestion model: {questioner_name}")
        print_separator()
        
        # Create results directory for the current round
        round_dir = os.path.join(experiment_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # 1. Question model generates a problem
        # Only display model name, not description
        print(f"{questioner_name} is generating a programming problem...")
        question_prompt = format_programming_prompt_question()
        question_response = questioner.call_api(question_prompt)
        
        # Parse question response
        question_data = extract_json_response(question_response)
        if not question_data:
            print("Failed to parse the question data, skipping this round")
            continue
        
        # Normalize question data
        normalized_data = normalize_question_data(question_data)
        
        problem = normalized_data["Problem"]
        standard_solution = normalized_data["Standard Solution"]
        scoring_standard = normalized_data["Scoring Standard"]
        
        # Alternative extraction logic: if normalized data is still empty
        # Check if extracted content is empty
        if not problem or (isinstance(problem, str) and not problem.strip()):
            print("Warning: Normalized problem is empty, attempting direct extraction...")
            # Try to extract problem directly from the original response
            problem_indicators = [
                "Problem:", "Task:", "Challenge:", "Question:", "You are tasked with", 
                "In this problem", "Create a program", "Your task is", "You need to", 
                "编写一个程序", "问题描述", "编程任务"
            ]
            
            for indicator in problem_indicators:
                if indicator in question_response:
                    start_idx = question_response.find(indicator) + len(indicator)
                    # Find a possible end point for the problem
                    end_markers = ["Standard Solution:", "Solution:", "Implementation:", "Scoring", "\n\n\n", "标准解决方案", "解决方案", "评分标准"]
                    end_positions = [question_response.find(marker, start_idx) for marker in end_markers]
                    valid_positions = [pos for pos in end_positions if pos > start_idx]
                    
                    if valid_positions:
                        end_idx = min(valid_positions)
                        problem = question_response[start_idx:end_idx].strip()
                        print(f"Directly extracted problem (first 100 chars): {problem[:100]}...")
                        break
            
            # If still empty, use the first 1000 characters of the response as the problem
            if not problem or (isinstance(problem, str) and not problem.strip()):
                print("Still could not extract problem, using first part of response...")
                problem = question_response[:1000].strip()
        
        # Check if standard solution is empty
        if not standard_solution or (isinstance(standard_solution, str) and not standard_solution.strip()):
            print("Warning: Normalized standard solution is empty, attempting direct extraction...")
            solution_indicators = [
                "Standard Solution:", "Solution:", "Implementation:", "Code:", "Here's the solution:",
                "标准解决方案:", "解决方案:", "代码实现:", "以下是解决方案:"
            ]
            
            for indicator in solution_indicators:
                if indicator in question_response:
                    start_idx = question_response.find(indicator) + len(indicator)
                    # Find a possible end point for the solution
                    end_markers = ["Scoring Standard:", "Scoring:", "Evaluation:", "Criteria:", "\n\n\n", "评分标准:", "评估标准:"]
                    end_positions = [question_response.find(marker, start_idx) for marker in end_markers]
                    valid_positions = [pos for pos in end_positions if pos > start_idx]
                    
                    if valid_positions:
                        end_idx = min(valid_positions)
                        standard_solution = question_response[start_idx:end_idx].strip()
                        print(f"Directly extracted solution (first 100 chars): {standard_solution[:100]}...")
                        break
            
            # If still empty, try to extract code blocks
            if not standard_solution or (isinstance(standard_solution, str) and not standard_solution.strip()):
                code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', question_response, re.DOTALL)
                if code_blocks:
                    standard_solution = max(code_blocks, key=len)
                    print(f"Extracted solution from code block (first 100 chars): {standard_solution[:100]}...")
        
        # Check if scoring standard is empty
        if not scoring_standard or (isinstance(scoring_standard, str) and not scoring_standard.strip()):
            print("Warning: Normalized scoring standard is empty, attempting direct extraction...")
            scoring_indicators = [
                "Scoring Standard:", "Scoring:", "Evaluation:", "Criteria:", "Grading:",
                "评分标准:", "评估标准:", "评分细则:", "打分标准:"
            ]
            
            for indicator in scoring_indicators:
                if indicator in question_response:
                    start_idx = question_response.find(indicator) + len(indicator)
                    # Find the end point for the scoring standard (usually the end of the response)
                    end_idx = len(question_response)
                    scoring_standard = question_response[start_idx:end_idx].strip()
                    print(f"Directly extracted scoring standard (first 100 chars): {scoring_standard[:100]}...")
                    break
            
            # If still empty, use default scoring standard
            if not scoring_standard or (isinstance(scoring_standard, str) and not scoring_standard.strip()):
                scoring_standard = """
                Correctness (40 points): Does the code correctly implement the required functionality?
                Efficiency (20 points): Is the code optimized in terms of time and space complexity?
                Code Quality (20 points): Is the code well-structured, readable, and following best practices?
                Completeness (20 points): Does the solution handle all edge cases and requirements?
                """
                print("Using default scoring standard")
        
        # Save question information to file
        final_question_data = {
            "Problem": problem,
            "Standard Solution": standard_solution,
            "Scoring Standard": scoring_standard,
            "Original": question_data  # Keep original data for subsequent analysis
        }
        with open(os.path.join(round_dir, "question.json"), "w", encoding="utf-8") as f:
            json.dump(final_question_data, f, ensure_ascii=False, indent=2)
        
        # Print problem, standard answer, and scoring standard in terminal
        print("\n[problem]")
        print(problem)
        print("\n[scoring standard]")
        print(scoring_standard)
        print("\n[standard solution]")
        print(standard_solution)
        print_separator()
        
        # 2. Other models answer the question
        answers = {}
        answering_models = [m for m in models if m != questioner]
        
        for model in answering_models:
            # Only display model name, not description
            model_name = model.name.split()[0] if ' ' in model.name else model.name
            print(f"{model_name} is answering the question...")
            answer_prompt = format_programming_prompt_answer(problem)
            answer_response = model.call_api(answer_prompt)
            
            # Parse answer
            answer_data = extract_json_response(answer_response)
            if answer_data:
                answer = answer_data.get("Answer", "Failed to parse the answer")
            else:
                answer = "Failed to parse the answer"
                
            answers[model.name] = answer
            
            # Print model answer, only use pure model name
            print(f"\n[{model_name}_answer]")
            print(answer)
            print("-" * 80)
            
            # Save answer
            with open(os.path.join(round_dir, f"answer_{model.name}.json"), "w", encoding="utf-8") as f:
                json.dump({"model": model.name, "answer": answer}, f, ensure_ascii=False, indent=2)
        
        print_separator()
        
        # 3. All models evaluate answers (questioner evaluates all answers, responder evaluates answers except their own)
        evaluations = {}
        
        # First, let the questioner evaluate all answers
        print(f"{questioner_name} (questioner) is evaluating all answers...")
        evaluate_prompt = format_programming_prompt_evaluate(problem, standard_solution, scoring_standard, answers, None)
        evaluation_response = questioner.call_api(evaluate_prompt)
        
        # Parse evaluation results
        evaluation_data = extract_json_response(evaluation_response)
        if evaluation_data:
            # Print scores and reasons, using pure model name
            print(f"\n[{questioner_name} (questioner) evaluation results]")
            
            if "Scores" in evaluation_data:
                print("Scores:")
                scores = evaluation_data["Scores"]
                for model_name, score in scores.items():
                    normalized_name = normalize_model_name(model_name)
                    if normalized_name and normalized_name in [m.name for m in models]:
                        print(f"{normalized_name}: {score}")
            
            if "Reason" in evaluation_data:
                print("\nEvaluation reason:")
                for model_name, reason in evaluation_data["Reason"].items():
                    # Only display model name, not description
                    short_name = model_name.split()[0] if ' ' in model_name else model_name
                    print(f"- {short_name}: {reason}")
                    print()
            
            print("-" * 40)
                
            evaluations[questioner.name] = evaluation_data
            
            # Save evaluation results
            with open(os.path.join(round_dir, f"evaluation_{questioner.name}.json"), "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        else:
            print(f"Can not parse the evaluation results of {questioner_name}")
        
        # Other models evaluate answers except their own
        for evaluator in answering_models:
            # Only display model name, not description
            evaluator_name = evaluator.name.split()[0] if ' ' in evaluator.name else evaluator.name
            print(f"{evaluator_name} is evaluating all answers except for itself...")
            evaluate_prompt = format_programming_prompt_evaluate(problem, standard_solution, scoring_standard, answers, evaluator.name)
            evaluation_response = evaluator.call_api(evaluate_prompt)
            
            # Parse evaluation results
            evaluation_data = extract_json_response(evaluation_response)
            if not evaluation_data:
                print(f"Can not parse the evaluation results of {evaluator_name}")
                continue
            
            # Print scores and reasons, using pure model name
            print(f"\n[{evaluator_name} evaluation results]")
            
            if "Scores" in evaluation_data:
                print("Scores:")
                scores = evaluation_data["Scores"]
                for model_name, score in scores.items():
                    normalized_name = normalize_model_name(model_name)
                    if normalized_name and normalized_name in [m.name for m in models]:
                        print(f"{normalized_name}: {score}")
            
            if "Reason" in evaluation_data:
                print("\nEvaluation reason:")
                for model_name, reason in evaluation_data["Reason"].items():
                    # Only display model name, not description
                    short_name = model_name.split()[0] if ' ' in model_name else model_name
                    print(f"- {short_name}: {reason}")
                    print()
            
            print("-" * 40)
                
            evaluations[evaluator.name] = evaluation_data
            
            # Save evaluation results
            with open(os.path.join(round_dir, f"evaluation_{evaluator.name}.json"), "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print_separator()
        
        # Normalize model names in all evaluation results
        for evaluator_name, evaluation in evaluations.items():
            if "Scores" not in evaluation:
                continue
            
            # Normalize scores in the model names
            normalized_scores = {}
            for model_name, score in evaluation["Scores"].items():
                normalized_name = normalize_model_name(model_name)
                if normalized_name and normalized_name in [m.name for m in models]:
                    normalized_scores[normalized_name] = score
            
            evaluation["Scores"] = normalized_scores
            
            # Normalize model names in reasons
            if "Reason" in evaluation:
                normalized_reason = {}
                for model_name, reason in evaluation["Reason"].items():
                    std_name = normalize_model_name(model_name)
                    if std_name and std_name in [m.name for m in models]:
                        normalized_reason[std_name] = reason
                evaluation["Reason"] = normalized_reason
        
        # Summarize current round results
        round_summary = {
            "round": round_num,
            "questioner": questioner.name,
            "problem": problem,
            "standard_solution": standard_solution,
            "scoring_standard": scoring_standard,
            "answers": answers,
            "evaluations": evaluations
        }
        
        with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
            json.dump(round_summary, f, ensure_ascii=False, indent=2)
            
        # Calculate initial scores
        initial_scores = {}
        score_details = {}  # Record scores given by each evaluator
        
        # Use scores directly from evaluators
        for evaluator_name, evaluation in evaluations.items():
            if "Scores" not in evaluation:
                continue
            
            # Initialize scores for the evaluator
            score_details[evaluator_name] = {}
            
            for model_name, score in evaluation["Scores"].items():
                normalized_name = normalize_model_name(model_name)
                if normalized_name and normalized_name in [m.name for m in models]:
                    initial_scores.setdefault(normalized_name, 0)
                    # Save to details
                    score_details[evaluator_name][normalized_name] = score
                    # Accumulate to total score
                    initial_scores[normalized_name] += score

        # Calculate normalized scores: total score divided by number of evaluators
        total_evaluators = len(models) - 1  # Excluding the questioner
        round_scores = {}
        for model_name, score in initial_scores.items():
            # Calculate normalized score: total score divided by number of evaluators
            normalized_score = (score / total_evaluators)
            round_scores[model_name] = normalized_score
        
        # Print round scores
        print(f"\n[Round {round_num} score]")
        print("-" * 120)
        # Print header: model name + each evaluator + total score
        header = "Model name".ljust(35)
        for evaluator_name in score_details.keys():
            # Use pure model name as column title
            short_name = evaluator_name.split()[0] if ' ' in evaluator_name else evaluator_name
            header += f"{short_name:<10}"
        header += "This round Total Score"
        print(header)
        print("-" * 120)
        
        # Sort by initial scores
        for model_name, total_score in sorted(initial_scores.items(), key=lambda x: x[1], reverse=True):
            # Only display model name, not description
            short_name = model_name.split()[0] if ' ' in model_name else model_name
            row = f"{short_name:<35}"
            # Add scores for each evaluator
            for evaluator_name, scores in score_details.items():
                model_score = scores.get(model_name, 0)
                row += f"{model_score:<10.1f}"
            
            # Add total score
            row += f"{total_score:<10.1f}"
            print(row)
        
        print("-" * 120)
        
        # Update total scores for each round
        for model_name, score in round_scores.items():
            if model_name in total_scores_by_round:
                total_scores_by_round[model_name].append(score)
        
        # Calculate current overall scores
        current_overall_scores = {}
        for model_name, scores in total_scores_by_round.items():
            if scores:  # Ensure there are score records
                # Scores are already in percentage, just take average
                current_overall_scores[model_name] = sum(scores) / round_num
            else:
                current_overall_scores[model_name] = 0
        
        # Calculate current ranking
        current_ranking = get_current_ranking(current_overall_scores)
        
        # Save round scores information
        round_score_data = {
            "round": round_num,
            "initial_scores": initial_scores,
            "score_details": score_details,  # Add score details
            "round_scores": round_scores,
            "current_overall_scores": current_overall_scores,  # Current overall scores
            "current_ranking": [(rank, name, score) for rank, name, score, _ in current_ranking]
        }
        all_round_scores.append(round_score_data)
        
        # Display total ranking
        display_total_ranking(current_ranking)
        
        # Save scores information to file
        with open(os.path.join(round_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(round_score_data, f, ensure_ascii=False, indent=2)
            
        # Display using pure model names
        questioner_name = questioner.name.split()[0] if ' ' in questioner.name else questioner.name
        print(f"======= The {round_num} round experiment is completed =======")
        
    # Experiment completed, summarize all rounds
    print_separator()
    print("All rounds of experiments have been completed, generating a summary report...")
    all_rounds = []
    
    for round_num in range(1, len(models) + 1):
        round_summary_path = os.path.join(experiment_dir, f"round_{round_num}", "round_summary.json")
        if os.path.exists(round_summary_path):
            with open(round_summary_path, "r", encoding="utf-8") as f:
                round_summary = json.load(f)
                all_rounds.append(round_summary)
    
    # Calculate final overall scores
    final_overall_scores = {}
    total_rounds = len(all_rounds)
    for model_name, scores in total_scores_by_round.items():
        if scores:  # Ensure there are score records
            # Scores are already in percentage, just take average
            final_overall_scores[model_name] = sum(scores) / total_rounds
        else:
            final_overall_scores[model_name] = 0
                
    # Add final scores and ranking information
    final_ranking = get_current_ranking(final_overall_scores)
    
    experiment_summary = {
        "experiment_time": timestamp,
        "total_rounds": len(all_rounds),
        "models": [model.name for model in models],
        "rounds": all_rounds,
        "scores_by_round": total_scores_by_round,  # Scores per round
        "final_overall_scores": final_overall_scores,  # Final overall scores
        "final_ranking": [(rank, name, score) for rank, name, score, _ in final_ranking],
        "round_scores": all_round_scores
    }
    
    with open(os.path.join(experiment_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
        
    print(f"Experiment report has been generated: {os.path.join(experiment_dir, 'experiment_summary.json')}")
    
    # Display final ranking
    print("\n======= Experiment Final Ranking =======")
    display_total_ranking(final_ranking)
    
    # Output Excel report
    print("\nGenerating Excel format report...")
    generate_excel_report(experiment_dir, models, all_round_scores, final_ranking, timestamp)
    print(f"Excel report has been generated: {os.path.join(experiment_dir, f'programming_experiment_{timestamp}.xlsx')}")

def generate_excel_report(experiment_dir, models, all_round_scores, final_ranking, timestamp):
    """Generate Excel format experiment report"""
    # Create Excel writer
    excel_path = os.path.join(experiment_dir, f"programming_experiment_{timestamp}.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # 1. Create each round scores table
    for round_data in all_round_scores:
        round_num = round_data["round"]
        
        # Prepare data
        model_names = []
        evaluator_columns = []
        round_scores = []
        total_scores = []
        
        # Find all evaluators
        evaluators = list(round_data["score_details"].keys())
        
        # Prepare data rows for each model
        for model_name, total_score in sorted(round_data["initial_scores"].items(), key=lambda x: x[1], reverse=True):
            short_name = model_name.split()[0] if ' ' in model_name else model_name
            model_names.append(short_name)
            
            # Add scores for each evaluator
            model_scores = []
            for evaluator_name in evaluators:
                # Get the score given by the evaluator for this model
                evaluator_scores = round_data["score_details"].get(evaluator_name, {})
                score = evaluator_scores.get(model_name, 0)
                model_scores.append(score)
            
            evaluator_columns.append(model_scores)
            round_scores.append(round_data["round_scores"].get(model_name, 0))
            total_scores.append(round_data["current_overall_scores"].get(model_name, 0))
        
        # Prepare DataFrame
        data = {'Model name': model_names}
        
        # Add evaluator columns
        for i, evaluator in enumerate(evaluators):
            short_evaluator = evaluator.split()[0] if ' ' in evaluator else evaluator
            data[f"{short_evaluator}"] = [scores[i] for scores in evaluator_columns]
        
        # Add round score and total score columns
        data['Round Score'] = round_scores
        data['Total Score'] = total_scores
        
        # Create DataFrame and write to Excel
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"Round {round_num}", index=False)
    
    # 2. Create final ranking table
    final_data = []
    for rank, model_name, score, is_tied in final_ranking:
        short_name = model_name.split()[0] if ' ' in model_name else model_name
        rank_display = f"{rank}{'*' if is_tied else ''}"  # Tie ranking marker
        final_data.append({
            'Rank': rank_display,
            'Model name': short_name,
            'Total Score': score
        })
    
    final_df = pd.DataFrame(final_data)
    final_df.to_excel(writer, sheet_name="Final Ranking", index=False)
    
    # 3. Create round scores summary table
    # Extract all model names
    all_model_names = [model.name for model in models]
    short_model_names = [name.split()[0] if ' ' in name else name for name in all_model_names]
    
    # Prepare summary data
    summary_data = {'Model name': short_model_names}
    max_rounds = len(all_round_scores)
    
    # Add round scores columns
    for round_num in range(1, max_rounds+1):
        round_data = next((r for r in all_round_scores if r["round"] == round_num), None)
        if round_data:
            round_scores = []
            for model_name in all_model_names:
                score = round_data["round_scores"].get(model_name, 0)
                round_scores.append(score)
            summary_data[f'Round {round_num}'] = round_scores
    
    # Add total score column
    total_scores = []
    for model_name in all_model_names:
        # Find the cumulative total score from the last round
        last_round = all_round_scores[-1]
        score = last_round["current_overall_scores"].get(model_name, 0)
        total_scores.append(score)
    summary_data['Total Score'] = total_scores
    
    # Create DataFrame and write to Excel
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name="Score Summary", index=False)
    
    # Save Excel file
    writer.close()

if __name__ == "__main__":
    run_programming_experiment() 
"""
Math Experiment
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

def format_math_prompt_question():
    """Construct the prompt for asking math questions"""
    return """You're a mathematician. Please perform the following tasks in order, and output only the final result in JSON format (no Markdown, no extra text):
1.Select a mathematical problem that you consider one of the most difficult. This can come from any field (e.g., algebra, geometry, number theory, etc.).
2.Identify the core mathematical principle(s) that make this problem difficult.
3.Create a new, original, and difficult mathematical problem based on the identified core principle(s).This problem must be entirely novel and not derived from any existing textbook or exam. 
4.Ensure that the new problem reflects both the core difficulty and the unique characteristics of the math domain.  
5.Solve the problem by providing a complete, step-by-step standard solution.
6.IMPORTANT: Write all mathematical equations in plain text format. DO NOT use LaTeX formatting or special symbols. For example:
   - Correct: "Find all integer solutions (x, y, z) to the equation x^2 + y^2 + z^2 = x^2*y^2"
   - Incorrect: "Find all integer solutions (x, y, z) to the equation $x^2 + y^2 + z^2 = x^2y^2$"
7.Output the final result only in JSON format, using the structure below:
{
  "Problem": "Your newly created difficult math problem here with equations in plain text.",
  "Standard Answer": "Step-by-step and final answer with explanation using plain text equations.",
  "Principle": "The abstracted mathematical principle that underlies the problem."
}
"""

def format_math_prompt_answer(problem):
    """Construct the prompt for answering math problems"""
    return f"""Please perform the following tasks:
1.First, think through the problem step by step. You may output your full reasoning process before the final answer.
2.Then, produce a final answer that is:
Concise, logically sound, and self-contained
final answer with necessary steps
Strictly no more than 500 words
Returned only as a valid JSON object with the key "Answer"
3.Important Constraints:
Only the final answer should appear in the JSON object.
Do not include any of your reasoning or thought process inside the JSON object.
The JSON object must be the last part of your output.
Do not use Markdown or any other formatting. Output should be plain text and a valid JSON object at the end.
IMPORTANT: Write all mathematical equations in plain text format. DO NOT use LaTeX formatting or special symbols. For example, write "x^2 + y = z" instead of "$x^2 + y = z$".
Problem:{problem}
Expected Final Output Format (JSON only):
{{
  "Answer": "Your final answer with necessary steps, under 500 words, goes here. Do not include reasoning or any text other than the final answer in this object. Use plain text for all equations."
}}
"""

def format_math_prompt_evaluate(problem, standard_answer, principle, all_answers, evaluator_name):
    """Construct the prompt for evaluating answers"""
    # Create all answers excluding the evaluator's own answer
    if evaluator_name is None:
        # Questioner evaluates all answers
        filtered_answers_str = json.dumps(all_answers, ensure_ascii=False, indent=2)
    else:
        # Answerer evaluates all answers except their own
        filtered_answers = {k: v for k, v in all_answers.items() if k != evaluator_name}
        filtered_answers_str = json.dumps(filtered_answers, ensure_ascii=False, indent=2)
    
    return f"""Evaluate the responses from multiple large language models based on the following information:
Problem:
{problem}
Standard Answer:
{standard_answer}
Principle:
{principle}
All the Models Answers (excluding your own):
{filtered_answers_str}
Please perform the following tasks:
1.Evaluate other models' answers based on the standard solution, considering  correctness, completeness, clarity, logical rigor and alignment with the given principle.
2.Rank all model responses from best to worst based on your evaluation.Please note that standard answers are not counted in the rankings and scores.
3.If any model answer is better than the standard answer, explicitly state this in your rationale and reflect it in the ranking.
4.IMPORTANT: Write all mathematical equations in plain text format. DO NOT use LaTeX formatting or special symbols. For example, write "x^2 + y = z" instead of "$x^2 + y = z$".
5.Output your result only in JSON format, with the following structure:
{{
  "Ranking": ["Model1", "Model2", "Model3", "..."],
  "Reason": {{
    "Model1": "Detailed explanation of strengths and weaknesses, including comparison to standard answer.",
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
        
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Handle LaTeX formulas and special characters in the response
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
        # If processing fails, continue with original response
        cleaned_response = response
    
    # First try to find JSON in the processed response
    try:
        # Look for the outermost pair of braces
        start = cleaned_response.find('{')
        end = cleaned_response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = cleaned_response[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, try to fix common JSON formatting issues
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                # Try to fix unescaped quotes
                json_str = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', json_str)
                # Try parsing again
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # Try to continue processing with the original response, looking for clearly marked answers
    try:
        # Look for the outermost pair of braces
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, try to see if it's a formatting issue
                # Remove all newlines and extra spaces
                json_str = re.sub(r'\s+', ' ', json_str).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # Look for JSON blocks at the end - usually preceded by "Final Answer:" or similar text
    try:
        # Look for ending JSON blocks
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
    
    # Try using regex to find JSON objects
    try:
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.finditer(json_pattern, response)
        for match in matches:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                # Verify if it contains expected fields
                if any(key in result for key in ["Answer", "Ranking", "Problem", "Standard Answer"]):
                    return result
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # Manually extract key information
    try:
        # Try to extract Ranking and Reason
        ranking = []
        reasons = {}
        
        # Look for the Ranking section
        if "Ranking" in response:
            ranking_start = response.find("Ranking") + len("Ranking") + 2  # +2 to skip ": ["
            ranking_end = response.find("]", ranking_start)
            if ranking_start > 0 and ranking_end > ranking_start:
                ranking_str = response[ranking_start:ranking_end]
                ranking = [item.strip().strip('"\'') for item in ranking_str.split(',')]
        
        # Look for the Reason section
        if "Reason" in response:
            reason_section = response[response.find("Reason"):]
            model_pattern = r'"([^"]+)":\s*"([^"]+)"'
            for match in re.finditer(model_pattern, reason_section):
                model_name = match.group(1)
                reason = match.group(2)
                reasons[model_name] = reason
        
        if ranking or reasons:
            result = {}
            if ranking:
                result["Ranking"] = ranking
            if reasons:
                result["Reason"] = reasons
            return result
    except Exception:
        pass
    
    # Check for "Answer" keyword
    try:
        # More flexibly detect Answer format
        patterns = [
            r'"Answer"\s*:\s*"([^"]+)"',  # Standard JSON format
            r'"Answer"\s*:\s*([^",\}\s][^",\}]*)',  # Value without quotes
            r'Answer:\s*([^\n]+)',  # Key-value pair but not in JSON
            r'答案:\s*([^\n]+)',  # Chinese format
            r'answer is\s*:?\s*([^\n,.]+)',  # Natural language statement
            r'(?:conclusion|总结)[\s:]*([^\n]{10,})'  # Conclusion section may contain the answer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                return {"Answer": answer_text}
    except Exception:
        pass
    
    # Try to extract answers from text containing LaTeX formulas
    try:
        # If response contains LaTeX formulas and "Answer"
        if (r"\[" in response or r"\(" in response or "$" in response) and "Answer" in response:
            # Look for the start of JSON block
            json_start = response.rfind("{", 0, response.rfind("}"))
            if json_start != -1:
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                try:
                    return json.loads(json_str)
                except:
                    # If JSON block cannot be parsed, try to extract the last line enclosed in braces with Answer in it
                    json_lines = [line for line in response.split('\n') if '{' in line and '}' in line and 'Answer' in line]
                    if json_lines:
                        # Extract the last line containing Answer
                        answer_line = json_lines[-1]
                        # Extract the JSON part
                        json_part = answer_line[answer_line.find('{'):answer_line.rfind('}')+1]
                        try:
                            return json.loads(json_part)
                        except:
                            pass
    except Exception:
        pass
    
    # Try to extract dictionary with problem, answer and principle
    try:
        # Check if it contains required keywords
        if "Problem" in response and "Standard Answer" in response and "Principle" in response:
            # Extract content to manually build JSON
            problem_start = response.find("Problem") + len("Problem") + 1
            problem_end = response.find("Standard Answer") - 1
            problem = response[problem_start:problem_end].strip().strip('":,')
            
            answer_start = response.find("Standard Answer") + len("Standard Answer") + 1
            answer_end = response.find("Principle") - 1
            standard_answer = response[answer_start:answer_end].strip().strip('":,')
            
            principle_start = response.find("Principle") + len("Principle") + 1
            principle = response[principle_start:].strip().strip('":,')
            
            return {
                "Problem": problem,
                "Standard Answer": standard_answer,
                "Principle": principle
            }
    except Exception:
        pass
    
    # If it's an answer case, try to extract the last few lines as the answer
    if "answer" in response.lower() or "答案" in response or "conclusion" in response.lower() or "总结" in response:
        try:
            # Get the last few lines
            lines = response.split('\n')
            # Look for lines containing JSON
            json_lines = [line for line in lines if '{' in line and '}' in line]
            if json_lines:
                for line in reversed(json_lines):  # Start from the end
                    try:
                        json_start = line.find('{')
                        json_end = line.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = line[json_start:json_end]
                            return json.loads(json_str)
                    except:
                        continue
            
            # If no JSON lines found, try to extract the last few lines as the answer
            last_lines = [line for line in lines[-7:] if len(line.strip()) > 10]  # Ensure line content is long enough
            if last_lines:
                return {"Answer": last_lines[-1].strip()}
        except Exception:
            pass
    
    # Print part of the response for debugging
    print(f"Unable to extract JSON from response, here is the original response")
    print("-" * 50)
    print(response[:500])
    print("-" * 50)
    
    # Last attempt - return the original text as Answer
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
        return None  # Return None to indicate this is not a valid model name
        
    # Convert to lowercase for matching
    name_lower = name.lower()
    
    # Get all model names and descriptions from configuration
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
    
    # Then add aliases and abbreviations, avoiding confusion from generic names
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
                  
        # O3-mini models
        elif "o3-mini" in model_name:
            name_mapping["o3-mini"] = std_name
            name_mapping["o3mini"] = std_name
            name_mapping["o3"] = std_name
            
        # O1 models
        elif "o1" == model_name:
            name_mapping["o1"] = std_name
            
        # Gemini models
        elif "gemini" in model_name:
            name_mapping["gemini-pro"] = std_name
            name_mapping["gemini-2.5"] = std_name
            name_mapping["gemini2.5"] = std_name
            
    # Try exact match first
    if name_lower in name_mapping:
        return name_mapping[name_lower]
    
    # Try matching with names and descriptions
    for model_name, (std_name, desc) in known_models.items():
        # Check if found in description or name
        if name_lower in model_name or (desc and name_lower in desc.lower()):
            return std_name
    
    # Try partial matching, but be more strict
    # First try to find matches containing specific version numbers
    for key, value in name_mapping.items():
        if (key in name_lower) and any(version_marker in key for version_marker in ["v3", "r1", "4o", "3-7", "2.5", "4v", "mini"]):
            return value
    
    # Other common abbreviations matching, but avoid confusion
    if "deepseek" in name_lower:
        # If version information is included, try exact matching
        if "v3" in name_lower or "coder" in name_lower:
            for model_name, (std_name, _) in known_models.items():
                if "deepseek-v3" == model_name:
                    return std_name
        elif "r1" in name_lower:
            for model_name, (std_name, _) in known_models.items():
                if "deepseek-r1" == model_name:
                    return std_name
        # If no clear version, don't make assumptions, return original name
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
            # Verify the model exists in MODELS
            for model_name, (std_name, _) in known_models.items():
                if value.lower() in model_name.lower():
                    return std_name
    
    # If no match found, return original name
    return name

def get_current_ranking(total_scores):
    """Calculate current ranking based on total scores"""
    # Sort by score in descending order
    sorted_models = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Handle tied scores
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
        # Only take the model name, not including description
        short_name = model_name.split()[0] if ' ' in model_name else model_name
        rank_display = f"{rank}{'*' if is_tied else ' '}"  # Mark tied rankings
        print(f"{rank_display:<5}{short_name:<35}{score:.2f}")
        
    print("-" * 80)
    print("Note: * indicates tied ranking")

def run_math_experiment():
    """Run math experiment"""
    # Initialize models
    models = init_llm()
    print(f"Successfully initialized {len(models)} models")
    
    # Create experiment results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"math_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize total score tracking
    total_scores_by_round = {model.name: [] for model in models}  # Store scores for each round
    all_round_scores = []
    
    # Each model takes turns asking questions
    for round_num, questioner in enumerate(models, 1):
        print_separator()
        print(f"======= Start of the {round_num} round experiment =======")
        # Only display model name, not description
        questioner_name = questioner.name.split()[0] if ' ' in questioner.name else questioner.name
        print(f"\nQuestion model: {questioner_name}")
        print_separator()
        
        # Create results directory for current round
        round_dir = os.path.join(experiment_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # 1. Questioning model generates a math problem
        # Only display model name, not description
        print(f"{questioner_name} is generating a math problem...")
        question_prompt = format_math_prompt_question()
        question_response = questioner.call_api(question_prompt)
        
        # Parse question response
        question_data = extract_json_response(question_response)
        if not question_data:
            print("Failed to parse the question data, skipping this round")
            continue
            
        problem = question_data.get("Problem", "")
        standard_answer = question_data.get("Standard Answer", "")
        principle = question_data.get("Principle", "")
        
        # Save question information to file
        with open(os.path.join(round_dir, "question.json"), "w", encoding="utf-8") as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)
        
        # Print problem, standard answer, and mathematical principle in terminal
        print("\n[problem]")
        print(problem)
        print("\n[principle]")
        print(principle)
        print("\n[standard_answer]")
        print(standard_answer)
        print_separator()
        
        # 2. Other models answer the question
        answers = {}
        answering_models = [m for m in models if m != questioner]
        
        for model in answering_models:
            # Only display model name, not description
            model_name = model.name.split()[0] if ' ' in model.name else model.name
            print(f"{model_name} is answering the question...")
            answer_prompt = format_math_prompt_answer(problem)
            answer_response = model.call_api(answer_prompt)
            
            # Parse answer
            answer_data = extract_json_response(answer_response)
            if answer_data:
                answer = answer_data.get("Answer", "Failed to parse the answer")
            else:
                answer = "Failed to parse the answer"
                
            answers[model.name] = answer
            
            # Print model's answer, only using pure model name
            print(f"\n[{model_name}_answer]")
            print(answer)
            print("-" * 80)  
            
            # Save answer
            with open(os.path.join(round_dir, f"answer_{model.name}.json"), "w", encoding="utf-8") as f:
                json.dump({"model": model.name, "answer": answer}, f, ensure_ascii=False, indent=2)
        
        print_separator()
        
        # 3. All models evaluate answers (questioner evaluates all answers, answerers evaluate all answers except their own)
        evaluations = {}
        
        # First let the questioner evaluate all answers
        print(f"{questioner_name} (questioner) is evaluating all answers...")
        evaluate_prompt = format_math_prompt_evaluate(problem, standard_answer, principle, answers, None)
        evaluation_response = questioner.call_api(evaluate_prompt)
        
        # Parse evaluation results
        evaluation_data = extract_json_response(evaluation_response)
        if evaluation_data:
            # Print ranking and reason, using pure model names
            print(f"\n[{questioner_name} (questioner) evaluation results]")
            
            if "Ranking" in evaluation_data:
                print("Ranking:")
                ranking = evaluation_data["Ranking"]
                # Standardize and filter rankings
                valid_ranking = []
                for model_name in ranking:
                    normalized_name = normalize_model_name(model_name)
                    if normalized_name and normalized_name in [m.name for m in models]:
                        valid_ranking.append(normalized_name)
                
                # Calculate scores: using a unified scoring standard
                max_score = len(models) - 2  # Total number of models minus 2 as highest score
                for i, model_name in enumerate(valid_ranking, 1):
                    # Only display model name, not description
                    short_name = model_name.split()[0] if ' ' in model_name else model_name
                    score = max(0, max_score - (i - 1))  # i starts from 1, so need to subtract 1
                    print(f"{i}. {short_name} {score} points")
            
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
        
        # Other models evaluate all answers except their own
        for evaluator in answering_models:
            # Only display model name, not description
            evaluator_name = evaluator.name.split()[0] if ' ' in evaluator.name else evaluator.name
            print(f"{evaluator_name} is evaluating all answers except itself...")
            evaluate_prompt = format_math_prompt_evaluate(problem, standard_answer, principle, answers, evaluator.name)
            evaluation_response = evaluator.call_api(evaluate_prompt)
            
            # Parse evaluation results
            evaluation_data = extract_json_response(evaluation_response)
            if not evaluation_data:
                print(f"Can not parse the evaluation results of {evaluator_name}")
                continue
            
            # Print ranking and reason, using pure model names
            print(f"\n[{evaluator_name} evaluation results]")
            
            if "Ranking" in evaluation_data:
                print("Ranking:")
                ranking = evaluation_data["Ranking"]
                # Standardize and filter rankings
                valid_ranking = []
                for model_name in ranking:
                    normalized_name = normalize_model_name(model_name)
                    if normalized_name and normalized_name in [m.name for m in models]:
                        valid_ranking.append(normalized_name)
                
                # Calculate scores: using a unified scoring standard
                max_score = len(models) - 2  # Total number of models minus 2 as highest score
                for i, model_name in enumerate(valid_ranking, 1):
                    # Only display model name, not description
                    short_name = model_name.split()[0] if ' ' in model_name else model_name
                    score = max(0, max_score - (i - 1))  # i starts from 1, so need to subtract 1
                    print(f"{i}. {short_name} {score} points")
            
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
        
        # Standardize model names in all evaluation results
        for evaluator_name, evaluation in evaluations.items():
            if "Ranking" not in evaluation:
                continue
            
            # Standardize model names in ranking, while filtering out invalid names
            normalized_ranking = []
            for model_name in evaluation["Ranking"]:
                normalized_name = normalize_model_name(model_name)
                if normalized_name and normalized_name in [m.name for m in models]:
                    normalized_ranking.append(normalized_name)
            
            evaluation["Ranking"] = normalized_ranking
            
            # Standardize model names in reason
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
            "standard_answer": standard_answer,
            "principle": principle,
            "answers": answers,
            "evaluations": evaluations
        }
        
        with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
            json.dump(round_summary, f, ensure_ascii=False, indent=2)
            
        # Calculate initial scores
        initial_scores = {}
        score_details = {}  # Record score details given by each evaluator
        
        # Use a unified scoring standard: total number of models minus 2 as the highest score
        max_score = len(models) - 2  
        
        # Scoring rules: rank first gets max_score points, second gets max_score-1 points, and so on
        for evaluator_name, evaluation in evaluations.items():
            if "Ranking" not in evaluation:
                continue
            
            ranking = evaluation["Ranking"]
            
            # Initialize evaluator's score details
            score_details[evaluator_name] = {}
            
            for i, model_name in enumerate(ranking):
                initial_scores.setdefault(model_name, 0)
                # Calculate the score given by the current evaluator for this model (highest score fixed at max_score)
                model_score = max(0, max_score - i)  # Ensure no negative scores
                # Save to details
                score_details[evaluator_name][model_name] = model_score
                # Accumulate to total score
                initial_scores[model_name] += model_score
                
        # Print first-level scoring results
        print(f"\n[Round {round_num} score]")
        print("-" * 120)
        # Print header: model name + each evaluator + total score
        header = "Model name".ljust(35)
        for evaluator_name in score_details.keys():
            # Use pure model name as column header
            short_name = evaluator_name.split()[0] if ' ' in evaluator_name else evaluator_name
            header += f"{short_name:<10}"
        header += "This round Total Score"
        print(header)
        print("-" * 120)
        
        # Use initial scores directly as the round score, but divide by (number of models - 1)
        total_evaluators = len(models) - 1  # Excluding the questioner
        round_scores = {}
        for model_name, score in initial_scores.items():
            # Calculate normalized score: total score divided by number of evaluators
            normalized_score = (score / total_evaluators) 
            round_scores[model_name] = normalized_score
        
        # Sort by initial scores
        for model_name, total_score in sorted(initial_scores.items(), key=lambda x: x[1], reverse=True):
            # Only display model name, not description
            short_name = model_name.split()[0] if ' ' in model_name else model_name
            row = f"{short_name:<35}"
            # Add score for each evaluator
            for evaluator_name, scores in score_details.items():
                model_score = scores.get(model_name, 0)
                row += f"{model_score:<10.1f}"
            
            # Add total score
            row += f"{total_score:<10.1f}"
            print(row)
        
        print("-" * 120)
        
        # Update round scores
        for model_name, score in round_scores.items():
            if model_name in total_scores_by_round:
                total_scores_by_round[model_name].append(score)
        
        # Calculate current overall scores
        current_overall_scores = {}
        for model_name, scores in total_scores_by_round.items():
            if scores:  # Ensure there are score records
                # Calculate average
                current_overall_scores[model_name] = sum(scores) / round_num
            else:
                current_overall_scores[model_name] = 0
        
        # Calculate current ranking
        current_ranking = get_current_ranking(current_overall_scores)
        
        # Save round score information
        round_score_data = {
            "round": round_num,
            "initial_scores": initial_scores,
            "score_details": score_details,  # Add score details
            "round_scores": round_scores,
            "current_overall_scores": current_overall_scores,  # Current overall score
            "current_ranking": [(rank, name, score) for rank, name, score, _ in current_ranking]
        }
        all_round_scores.append(round_score_data)
        
        # Display total ranking
        display_total_ranking(current_ranking)
        
        # Save score information to file
        with open(os.path.join(round_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(round_score_data, f, ensure_ascii=False, indent=2)
            
        # Display using pure model name
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
            # Calculate average
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
        "scores_by_round": total_scores_by_round,  # Scores for each round
        "final_overall_scores": final_overall_scores,  # Final overall score
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
    print(f"Excel report has been generated: {os.path.join(experiment_dir, f'math_experiment_{timestamp}.xlsx')}")

def generate_excel_report(experiment_dir, models, all_round_scores, final_ranking, timestamp):
    """Generate Excel format experiment report"""
    # Create Excel writer
    excel_path = os.path.join(experiment_dir, f"math_experiment_{timestamp}.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # 1. Create each round score table
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
            
            # Add score for each evaluator
            model_scores = []
            for evaluator_name in evaluators:
                # Get the evaluator's score for this model
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
        rank_display = f"{rank}{'*' if is_tied else ''}"  # Tie ranking mark
        final_data.append({
            'Rank': rank_display,
            'Model name': short_name,
            'Total Score': score
        })
    
    final_df = pd.DataFrame(final_data)
    final_df.to_excel(writer, sheet_name="Final Ranking", index=False)
    
    # 3. Create round score summary table
    # Extract all model names
    all_model_names = [model.name for model in models]
    short_model_names = [name.split()[0] if ' ' in name else name for name in all_model_names]
    
    # Prepare summary data
    summary_data = {'Model name': short_model_names}
    max_rounds = len(all_round_scores)
    
    # Add round score columns
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
        # Find the cumulative total score of the last round
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
    run_math_experiment() 
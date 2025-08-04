"""
Model access and calling module: Encapsulates API call related functionality
"""

import json
import requests
import time
import random
from config import API_BASE, STREAMING, DEFAULT_TEMPERATURE

class LLMModel:
    """Large Language Model wrapper class, handles API calls"""
    
    def __init__(self, name, description, api_key, streaming=None):
        self.name = name
        self.description = description
        self.api_key = api_key
        self.api_base = API_BASE
        # If a model-specific streaming setting is specified, use it, otherwise use the global setting
        self.streaming = STREAMING if streaming is None else streaming
    
    def __str__(self):
        return f"{self.name} ({self.description})"
        
    def call_api(self, prompt, temperature=DEFAULT_TEMPERATURE, stream=None, max_retries=6, retry_delay=3):
        """Call API to get model response"""
        # If stream parameter is not specified, use the model-specific streaming setting
        if stream is None:
            stream = self.streaming
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": stream
        }
        
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Implement exponential backoff strategy: waiting time for each retry increases exponentially
                    # Waiting time for nth retry: base delay * (3^n) + random value
                    backoff_time = retry_delay * (3 ** (attempt)) + random.uniform(3, 5)
                    print(f"Attempting to call API again, attempt {attempt+1}... (waiting {backoff_time:.2f} seconds)")
                    time.sleep(backoff_time)
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    stream=stream,
                    timeout=120  # Set timeout to 120 seconds
                )
                
                if stream:
                    # Handle streaming response
                    full_response = ""
                    try:
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith('data: '):
                                    chunk_data = chunk_str[6:]  # Remove 'data: ' prefix
                                    if chunk_data != "[DONE]":
                                        try:
                                            chunk_json = json.loads(chunk_data)
                                            choices = chunk_json.get('choices', [])
                                            if choices and len(choices) > 0:
                                                content = choices[0].get('delta', {}).get('content', '')
                                                if content:
                                                    full_response += content
                                                    print(content, end='', flush=True)
                                            else:
                                                print(f"Warning: API returned abnormal data format, choices not found or empty: {chunk_data[:100]}")
                                        except json.JSONDecodeError as e:
                                            print(f"Error parsing JSON chunk: {str(e)}")
                                            # Continue processing the next chunk
                        print()  # Add a newline at the end
                        
                        # Check if any response was collected
                        if not full_response:
                            print("No valid response collected, may need to retry")
                            if attempt < max_retries - 1:
                                continue
                            else:
                                return "No valid response collected"
                                
                        return full_response
                    except Exception as e:
                        print(f"Error processing streaming response: {str(e)}")
                        # If there is a partial response and this is the last attempt
                        if full_response and attempt == max_retries - 1:
                            print("Returning partially collected response...")
                            return full_response
                        # Otherwise continue retrying
                        continue
                else:
                    # Handle non-streaming response
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            content = result['choices'][0]['message']['content']
                            if not content:
                                print("API returned empty content")
                                if attempt < max_retries - 1:
                                    continue
                            return content
                        except (KeyError, json.JSONDecodeError) as e:
                            print(f"Error parsing API response: {str(e)}")
                            if attempt < max_retries - 1:
                                continue
                            return f"Error parsing API response: {str(e)}, original response: {response.text[:200]}"
                    else:
                        print(f"API call failed, status code: {response.status_code}")
                        print(f"Error message: {response.text}")
                        # If it's a 429 or 5xx error, try to retry
                        if (response.status_code == 429 or response.status_code >= 500) and attempt < max_retries - 1:
                            print(f"Server busy or error, will retry...")
                            # For 429 errors, add a longer waiting time
                            if response.status_code == 429:
                                time.sleep(retry_delay * 2 + random.uniform(0, 5))
                            continue
                        # If it's the last attempt and failed
                        if attempt == max_retries - 1:
                            return f"API call failed: status code {response.status_code}, response: {response.text[:200]}"
                
            except requests.exceptions.Timeout:
                print(f"API call timeout, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return "API call timeout"
            
            except requests.exceptions.RequestException as e:
                print(f"API call network error: {str(e)}, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return f"API call network error: {str(e)}"
            
            except Exception as e:
                print(f"API call unknown exception: {str(e)}, attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return f"API call unknown exception: {str(e)}"
        
        # If all retries fail
        return "All API calls failed, unable to get response" 
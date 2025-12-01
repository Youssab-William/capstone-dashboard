import asyncio
import aiohttp
import json
import csv
import os
import time
import datetime
from typing import List, Dict, Any
import nest_asyncio

# Apply the patch to allow asyncio to run in environments like Colab
nest_asyncio.apply()


# --- Configuration ---
# IMPORTANT: Replace "YOUR_DEEPSEEK_API_KEY" with your actual DeepSeek API key.
# You can also set this as an environment variable for better security.
API_KEY = "sk-4259c706c25940d9bbf946d0e4cd131a"
BASE_URL = "https://api.deepseek.com/chat/completions"

# --- Model and File Settings ---
MODEL_NAME = "deepseek-chat"  # Or "deepseek-reasoner"
INPUT_FILENAME = "additional_prompts.txt" # JSON-formatted prompts file
OUTPUT_JSON_FILENAME = "deepseek_additional_results.json"
OUTPUT_CSV_FILENAME = "deepseek_additional_results.csv"

# --- Concurrency Settings ---
# The number of prompts to process at the same time.
# Adjust this based on your needs and the API's rate limits.
# Start with a lower number (like 5 or 10) and increase if stable.
MAX_CONCURRENT_REQUESTS = 10

# --- Function to read prompts from a file ---
def get_prompts_from_file(filename: str) -> List[Dict[str, Any]]:
    """Reads prompts from a JSON-formatted text file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if the content is a JSON array
            if content.strip().startswith('['):
                prompts = json.loads(content)
            else:
                # If not a JSON array, try to parse each line as a separate JSON object
                prompts = [json.loads(line) for line in f.readlines() if line.strip()]
        if not prompts:
            print(f"Warning: Input file '{filename}' is empty or contains only whitespace.")
        return prompts
    except FileNotFoundError:
        print(f"Error: The input file '{filename}' was not found.")
        print("Please create this file with your prompts in JSON format.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: The input file '{filename}' is not valid JSON: {e}")
        return []

# --- Asynchronous function to process a single prompt ---
async def process_prompt(session: aiohttp.ClientSession, prompt_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Sends a single prompt to the DeepSeek API and returns the result.
    Uses a semaphore to limit concurrency.
    """
    task_id = prompt_data.get("TaskID", "Unknown")
    prompt_tone = prompt_data.get("PromptTone", "Unknown")
    prompt_text = prompt_data.get("PromptText", "")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "stream": False # We want the full response at once
    }

    # The 'async with' block ensures the semaphore is released even if an error occurs
    async with semaphore:
        print(f"Processing prompt for TaskID: {task_id}, Tone: {prompt_tone}")
        start_time = time.time()
        try:
            # Add a timeout to the request to prevent it from hanging indefinitely
            async with session.post(BASE_URL, headers=headers, json=payload, timeout=180) as response:
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                result = await response.json()
                # Extract the content from the response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "No content found.")
                
                # Extract token usage information
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                processing_time = time.time() - start_time
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return {
                    "TaskID": task_id,
                    "PromptTone": prompt_tone,
                    "Prompt": prompt_text,
                    "Model": MODEL_NAME,
                    "ResponseText": content,
                    "ResponseLength": completion_tokens,  # Using completion tokens instead of character length
                    "ProcessingTime": round(processing_time, 2),
                    "Timestamp": timestamp
                }
        except aiohttp.ClientError as e:
            print(f"An API error occurred for TaskID {task_id}: {e}")
            error_msg = f"Error: {e}"
            processing_time = time.time() - start_time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "TaskID": task_id,
                "PromptTone": prompt_tone,
                "Prompt": prompt_text,
                "Model": MODEL_NAME,
                "ResponseText": error_msg,
                "ResponseLength": 0,  # No tokens for error messages
                "ProcessingTime": round(processing_time, 2),
                "Timestamp": timestamp
            }
        except asyncio.TimeoutError:
            print(f"A timeout error occurred for TaskID {task_id}")
            error_msg = "Error: Request timed out."
            processing_time = time.time() - start_time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "TaskID": task_id,
                "PromptTone": prompt_tone,
                "Prompt": prompt_text,
                "Model": MODEL_NAME,
                "ResponseText": error_msg,
                "ResponseLength": 0,  # No tokens for error messages
                "ProcessingTime": round(processing_time, 2),
                "Timestamp": timestamp
            }
        except Exception as e:
            print(f"An unexpected error occurred for TaskID {task_id}: {e}")
            error_msg = f"Error: {e}"
            processing_time = time.time() - start_time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "TaskID": task_id,
                "PromptTone": prompt_tone,
                "Prompt": prompt_text,
                "Model": MODEL_NAME,
                "ResponseText": error_msg,
                "ResponseLength": 0,  # No tokens for error messages
                "ProcessingTime": round(processing_time, 2),
                "Timestamp": timestamp
            }

# --- Functions to save results ---
def save_to_json(data: List[Dict[str, Any]], filename: str):
    """Saves the list of results to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSuccessfully saved {len(data)} responses to {filename}")

def save_to_csv(data: List[Dict[str, Any]], filename: str):
    """Saves the list of results to a CSV file."""
    if not data:
        return # Don't create an empty file with only headers
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        # Define the CSV header based on the keys of the first dictionary
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Successfully saved {len(data)} responses to {filename}")

# --- Main asynchronous function ---
async def main():
    """
    Main function to orchestrate reading prompts, processing them concurrently,
    and saving the results.
    """
    if API_KEY == "YOUR_DEEPSEEK_API_KEY":
        print("Error: Please replace 'YOUR_DEEPSEEK_API_KEY' with your actual API key in the script.")
        return

    prompt_data = get_prompts_from_file(INPUT_FILENAME)
    if not prompt_data:
        print("Exiting script as there are no prompts to process.")
        return

    print(f"Found {len(prompt_data)} prompts to process.")

    # A semaphore to limit the number of concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Use a single aiohttp session for all requests for better performance
    async with aiohttp.ClientSession() as session:
        # Create a list of tasks to be executed concurrently
        tasks = [process_prompt(session, prompt, semaphore) for prompt in prompt_data]

        # asyncio.gather runs all tasks and collects their results
        results = await asyncio.gather(*tasks)

    # Filter out any potential None results if errors were not handled to return a dict
    valid_results = [res for res in results if res]

    # Save the results to both JSON and CSV formats
    if valid_results:
        save_to_json(valid_results, OUTPUT_JSON_FILENAME)
        save_to_csv(valid_results, OUTPUT_CSV_FILENAME)
    else:
        print("No valid results were generated.")

# --- Run the script ---
# Because nest_asyncio is used, we can now safely use asyncio.run()
# even in environments like Colab/Jupyter.
if __name__ == "__main__":
    asyncio.run(main())
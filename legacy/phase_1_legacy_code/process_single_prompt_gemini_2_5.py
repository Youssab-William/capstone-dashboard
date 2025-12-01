import google.generativeai as genai
import os
import time
import sys
import csv
from datetime import datetime

# Configure API key
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    # Fallback to hardcoded key (for testing only)
    API_KEY = "AIzaSyATxilMVX1MuCQbHgpcZoTaEk6O45lJLXY"  # Replace with your new API key
    print(f"Using hardcoded API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    print("WARNING: Using hardcoded API keys is not secure for production environments.")
    print("Consider setting the GEMINI_API_KEY environment variable instead.")

try:
    # Configure API key
    genai.configure(api_key=API_KEY)
    print("API configuration successful")
except Exception as e:
    print(f"Error configuring API: {e}")
    sys.exit(1)

# Function to process a single prompt with Gemini 2.5 Pro
def process_prompt(prompt_data):
    # Extract prompt information
    task_id = prompt_data.get("TaskID", "")
    prompt_tone = prompt_data.get("PromptTone", "")
    prompt_text = prompt_data.get("PromptText", "")
    
    print(f"\nProcessing prompt: {task_id}")
    print(f"Prompt tone: {prompt_tone}")
    print(f"Prompt text: {prompt_text[:50]}...")
    
    # Set retry delay
    retry_delay = 60  # Delay in seconds (1 minute)
    
    while True:
        try:
            # Initialize the model
            print("\nInitializing the model gemini-2.5-pro...")
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            # Record start time for processing time calculation
            start_time = time.time()
            
            # Send the prompt and get the response
            print(f"Sending prompt to Gemini 2.5 Pro...")
            response = model.generate_content(prompt_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract response text
            response_text = response.text
            
            # Get token count from response.usage_metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                # Extract token count from the response metadata
                candidates_token_count = getattr(response.usage_metadata, 'candidates_token_count', 0)
                response_length = candidates_token_count
                print(f"Response received (length: {response_length} tokens)")
                print(f"Token metadata: {response.usage_metadata}")
            else:
                # Fallback to character count if token count is not available
                response_length = len(response_text)
                print(f"Response received (length: {response_length} characters - token count not available)")
            
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Create result dictionary with only the requested columns
            result = {
                'TaskID': task_id,
                'PromptTone': prompt_tone,
                'Prompt': prompt_text,
                'Model': 'gemini-2.5-pro',
                'ResponseText': response_text,
                'ResponseLength': response_length,
                'ProcessingTime': f"{processing_time:.2f}",
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
        
        except Exception as e:
            print(f"Error processing prompt: {e}")
            
            # For any error, retry after delay
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            print("Retrying now...")
            continue

# Function to save results to a CSV file
def save_to_csv(results, output_file, append_mode=False):
    # Define the CSV columns - only the essential ones as requested
    columns = [
        'TaskID', 'PromptTone', 'Prompt', 'Model', 'ResponseText', 'ResponseLength',
        'ProcessingTime', 'Timestamp'
    ]
    
    # Write the results to the CSV file
    # Use append mode if specified, otherwise use write mode
    mode = 'a' if append_mode else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        # Only write header if not in append mode
        if not append_mode:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Path to the JSON file with prompts
    json_file = "additional_prompts.txt"
    
    # Output CSV file
    output_file = "additional_gemini_2_5_pro_results.csv"
    
    # Read the JSON file
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Read existing results to continue from where we left off
    existing_results = []
    try:
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results.append(row)
        print(f"Loaded {len(existing_results)} existing results from {output_file}")
    except FileNotFoundError:
        print(f"No existing results file found at {output_file}. Starting fresh.")
    
    # Start from the beginning since we're processing a new file
    start_index = 0  # 0-indexed, so prompt 1 is at index 0
    
    # Process remaining prompts
    if len(data) > 0:
        print(f"Found {len(data)} prompts to process")
        print(f"Starting from prompt {start_index + 1} (index {start_index})")
        
        # Process each prompt starting from index 91 (prompt 92)
        for i in range(start_index, len(data)):
            prompt_data = data[i]
            print(f"\nProcessing prompt {i+1} of {len(data)}")
            result = process_prompt(prompt_data)
            existing_results.append(result)
            
            # Create a list with just the new result
            new_result = [result]
            
            # Save only the new result in append mode
            save_to_csv(new_result, output_file, append_mode=True)
            print(f"New result appended to {output_file}")
            
            # Add a small delay between prompts to avoid rate limiting
            if i < len(data) - 1:  # Don't delay after the last prompt
                delay = 2  # 2 seconds delay between prompts
                print(f"Waiting {delay} seconds before processing next prompt...")
                time.sleep(delay)
        
        print(f"\nAll remaining prompts processed successfully")
        print(f"All results have been appended to {output_file}")
    else:
        print("No prompts found in the JSON file.")
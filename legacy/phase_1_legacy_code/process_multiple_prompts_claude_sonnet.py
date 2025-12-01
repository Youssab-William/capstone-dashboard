import anthropic
import json
import time
import os
import csv
import sys
from datetime import datetime
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Configure API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    # Fallback to asking for the key
    print("ANTHROPIC_API_KEY not found in environment variables.")
    ANTHROPIC_API_KEY = input("Please enter your Anthropic API key: ").strip()
    if not ANTHROPIC_API_KEY:
        print("No API key provided. Exiting.")
        sys.exit(1)
    print(f"Using provided API key: {ANTHROPIC_API_KEY[:5]}...{ANTHROPIC_API_KEY[-5:]}")
    print("WARNING: Consider setting the ANTHROPIC_API_KEY environment variable for future use.")

try:
    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("API configuration successful")
except Exception as e:
    print(f"Error configuring API: {e}")
    sys.exit(1)

# Function to estimate tokens in a text
def estimate_tokens(text):
    """Estimate the number of tokens in a text.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated number of tokens
    """
    # Claude uses approximately 4 characters per token on average
    # This is a rough estimate and may not be exact
    if not text:
        return 0
    return len(text) // 4

# Function to save results to a CSV file
def save_to_csv(results, output_file, append_mode=False):
    # Define the CSV columns
    fieldnames = [
        'TaskID', 'PromptTone', 'Prompt', 'Model', 'ResponseText', 'ResponseLength',
        'InputTokens', 'TotalTokens', 'ProcessingTime', 'Timestamp'
    ]
    
    # Write the results to the CSV file
    # Use append mode if specified, otherwise use write mode
    mode = 'a' if append_mode else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Only write header if not in append mode
        if not append_mode:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")

# Function to process prompts in batches
def process_prompts_in_batch(prompts, model="claude-sonnet-4-20250514", batch_size=10, max_tokens=2503):
    """Process multiple prompts in batches using the batch API.
    
    Args:
        prompts (list): List of prompt data dictionaries
        model (str): Model to use for processing
        batch_size (int): Number of prompts to process in each batch
        max_tokens (int): Maximum tokens for each response
        
    Returns:
        list: List of results
    """
    results = []
    total_prompts = len(prompts)
    
    # Process prompts in batches
    for i in range(0, total_prompts, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_size_actual = len(batch_prompts)
        
        print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size_actual, total_prompts)} of {total_prompts})")
        
        # Create batch requests
        requests = []
        for j, prompt_data in enumerate(batch_prompts):
            prompt_text = prompt_data.get("PromptText", "")
            task_id = prompt_data.get("TaskID", "")
            
            # Use TaskID as custom_id for tracking, but ensure uniqueness within batch
            # by appending the index within the batch
            custom_id = f"{task_id}-{j}" if task_id else f"prompt-{i+j+1}"
            
            requests.append(
                Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{
                            "role": "user",
                            "content": prompt_text
                        }]
                    )
                )
            )
        
        # Record start time for processing time calculation
        start_time = time.time()
        
        try:
            # Send batch request
            print(f"Sending batch request to {model}...")
            message_batch = client.messages.batches.create(requests=requests)
            
            # Wait for batch to complete
            batch_id = message_batch.id
            print(f"Batch ID: {batch_id}")
            print("Waiting for batch to complete...")
            
            while True:
                batch_status = client.messages.batches.retrieve(batch_id)
                print(f"Current status: {batch_status.processing_status}")
                
                if batch_status.processing_status == "ended":
                    break
                elif batch_status.processing_status in ["failed", "canceled"]:
                    print(f"Batch processing failed or was canceled: {batch_status.processing_status}")
                    raise Exception(f"Batch processing {batch_status.processing_status}")
                
                # Wait before checking again
                time.sleep(5)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"Batch processed in {processing_time:.2f} seconds")
            
            # Retrieve and process results
            batch_results = client.messages.batches.results(batch_id)
            
            for j, result_item in enumerate(batch_results):
                custom_id = result_item.custom_id
                prompt_data = next((p for p in batch_prompts if p.get("TaskID", "") == custom_id), batch_prompts[j])
                
                task_id = prompt_data.get("TaskID", "")
                prompt_tone = prompt_data.get("PromptTone", "")
                prompt_text = prompt_data.get("PromptText", "")
                
                if result_item.result.type == "succeeded":
                    # Get the message and usage information
                    message = result_item.result.message
                    usage = message.usage
                    
                    # Extract response text
                    response_text = message.content[0].text
                    
                    # Get exact token count from usage metadata
                    response_tokens = usage.output_tokens
                    
                    # Create result dictionary
                    result = {
                        'TaskID': task_id,
                        'PromptTone': prompt_tone,
                        'Prompt': prompt_text,
                        'Model': model,
                        'ResponseText': response_text,
                        'ResponseLength': response_tokens,
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                elif result_item.result.type == "errored":
                    # Handle error case
                    error_message = result_item.result.error.message if hasattr(result_item.result.error, 'message') else str(result_item.result.error)
                    
                    # Create error result dictionary
                    result = {
                        'TaskID': task_id,
                        'PromptTone': prompt_tone,
                        'Prompt': prompt_text,
                        'Model': model,
                        'ResponseText': f"Error: {error_message}",
                        'ResponseLength': 0,
                        'InputTokens': 0,
                        'TotalTokens': 0,
                        'ProcessingTime': f"{processing_time / batch_size_actual:.2f}",
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                results.append(result)
                
                # Print processing information
                if result_item.result.type == "succeeded":
                    print(f"Processed prompt {i+j+1} of {total_prompts} - Input tokens: {usage.input_tokens}, Output tokens: {response_tokens}, Total: {usage.input_tokens + usage.output_tokens}")
                else:
                    print(f"Processed prompt {i+j+1} of {total_prompts} - Error occurred")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            
            # Create error results for all prompts in the batch
            for j, prompt_data in enumerate(batch_prompts):
                task_id = prompt_data.get("TaskID", "")
                prompt_tone = prompt_data.get("PromptTone", "")
                prompt_text = prompt_data.get("PromptText", "")
                
                error_result = {
                    'TaskID': task_id,
                    'PromptTone': prompt_tone,
                    'Prompt': prompt_text,
                    'Model': model,
                    'ResponseText': f"Error: {str(e)}",
                    'ResponseLength': 0,
                    'InputTokens': 0,
                    'TotalTokens': 0,
                    'ProcessingTime': "0.00",
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                results.append(error_result)
                print(f"Recorded error for prompt {i+j+1} of {total_prompts}")
        
        # Print batch summary if batch_id was assigned (no error in batch creation)
        if 'batch_id' in locals():
            try:
                batch_final = client.messages.batches.retrieve(batch_id)
                if batch_final.request_counts:
                    print(f"\n--- Batch Summary ---")
                    print(f"Total requests: {batch_final.request_counts.processing + batch_final.request_counts.succeeded + batch_final.request_counts.errored}")
                    print(f"Succeeded: {batch_final.request_counts.succeeded}")
                    print(f"Errored: {batch_final.request_counts.errored}")
                    print(f"Processing: {batch_final.request_counts.processing}")
            except Exception as e:
                print(f"Error retrieving batch summary: {e}")
        
        # Add a delay between batches to avoid rate limiting
        if i + batch_size < total_prompts:  # Don't delay after the last batch
            delay = 5  # 5 seconds delay between batches
            print(f"Waiting {delay} seconds before processing next batch...")
            time.sleep(delay)
    
    return results

# Main execution
if __name__ == "__main__":
    # Path to the JSON file with prompts
    json_file = "additional_prompts.txt"
    
    # Output CSV file
    output_file = "claude_additional_results.csv"
    
    # Model to use
    model = "claude-sonnet-4-20250514"
    
    print(f"\nUsing model: {model}")
    print(f"Results will be saved to: {output_file}")
    
    # Ask for batch size
    try:
        batch_size_input = input("\nEnter batch size (default is 10, max 50): ")
        batch_size = int(batch_size_input) if batch_size_input.strip() else 10
        batch_size = min(max(1, batch_size), 50)  # Ensure between 1 and 50
    except ValueError:
        batch_size = 10
        print("Invalid input. Using default batch size of 10.")
    
    print(f"Using batch size: {batch_size}")
    
    # Set max tokens to 2503 (max from gemini_2_5_pro_all_results.csv)
    max_tokens = 2503
    print(f"Using max tokens: {max_tokens} (maximum from gemini_2_5_pro_all_results.csv)")
    
    print("\n=== Starting Prompt Processing ===\n")
    
    # Read the JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        sys.exit(1)
    
    # Read existing results to continue from where we left off
    existing_results = []
    try:
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results.append(row)
        print(f"Loaded {len(existing_results)} existing results from {output_file}")
        # Start from where we left off
        start_index = len(existing_results)
    except FileNotFoundError:
        print(f"No existing results file found at {output_file}. Starting fresh.")
        start_index = 0
    
    # Process remaining prompts
    if len(data) > 0:
        print(f"Found {len(data)} prompts to process")
        print(f"Starting from prompt {start_index + 1} (index {start_index})")
        
        # Get remaining prompts
        remaining_prompts = data[start_index:]
        
        if len(remaining_prompts) > 0:
            # Process prompts in batches
            results = process_prompts_in_batch(remaining_prompts, model, batch_size, max_tokens)
            
            # Save all results at once
            save_to_csv(results, output_file, append_mode=(start_index > 0))
            
            print(f"\nAll remaining prompts processed successfully")
            print(f"All results have been saved to {output_file}")
        else:
            print("No remaining prompts to process.")
    else:
        print("No prompts found in the JSON file.")
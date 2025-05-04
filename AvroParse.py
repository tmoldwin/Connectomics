import os
import pandas as pd
from tqdm import tqdm
import logging
import json

# Suppress verbose logging
logging.basicConfig(level=logging.ERROR)

data_folder = r'C:\Program Files (x86)\Google\Cloud SDK\synapses'  # UPDATE this path

# Configuration options
MAX_FILES_TO_PROCESS = -1  # Number of files to process
MAX_RECORDS_PER_FILE = 100000  # Maximum records per output file

# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Create output directory if it doesn't exist
output_dir = "synapse_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

total_records = 0
processed_files = 0
error_files = 0

# Get the list of files to process
files_to_process = [f for f in sorted(os.listdir(data_folder)) if f.endswith('.json')]
print(f"Found {len(files_to_process)} JSON schema files, will process first {MAX_FILES_TO_PROCESS} for testing")

# Limit the number of files to process
files_to_process = files_to_process[:MAX_FILES_TO_PROCESS]

# Create a progress bar
with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
    for file_index, filename in enumerate(files_to_process):
        json_path = os.path.join(data_folder, filename)
        records = []  # Reset records for each file

        try:
            # Process the JSON file directly
            with open(json_path, 'r') as json_file:
                for line in json_file:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        # Flatten the nested structure
                        flat_record = flatten_dict(record)
                        records.append(flat_record)
                    except json.JSONDecodeError:
                        continue
            
            # Save this file's data immediately
            if records:
                df = pd.DataFrame(records)
                total_records += len(df)
                
                # Split into multiple files if needed
                if len(df) > MAX_RECORDS_PER_FILE:
                    num_chunks = (len(df) + MAX_RECORDS_PER_FILE - 1) // MAX_RECORDS_PER_FILE
                    for i in range(num_chunks):
                        start_idx = i * MAX_RECORDS_PER_FILE
                        end_idx = min((i + 1) * MAX_RECORDS_PER_FILE, len(df))
                        chunk = df.iloc[start_idx:end_idx]
                        output_path = os.path.join(output_dir, f'file{file_index+1}_part{i+1}.csv')
                        chunk.to_csv(output_path, index=False)
                else:
                    output_path = os.path.join(output_dir, f'file{file_index+1}.csv')
                    df.to_csv(output_path, index=False)
                
                print(f"Saved {len(df)} records from {filename} with {len(df.columns)} columns")
                processed_files += 1
            else:
                print(f"No valid records found in {filename}")
                
        except Exception as e:
            error_files += 1
            print(f"Error processing {filename}: {str(e)[:100]}...")
        
        pbar.update(1)

print(f"\nSummary: Processed {total_records} total records from {processed_files} files")
print(f"Encountered errors in {error_files} files")
print(f"CSV files saved in: {os.path.abspath(output_dir)}")

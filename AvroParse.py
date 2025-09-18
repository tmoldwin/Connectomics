import os
import pandas as pd
from tqdm import tqdm
import logging
import json
import gc

# Suppress verbose logging
logging.basicConfig(level=logging.ERROR)

data_folder = r'C:\synapse_data'  # UPDATE this path

# Configuration options
MAX_FILES_TO_PROCESS = -1  # Process ALL files
CHUNK_SIZE = 10000  # Process in chunks to avoid memory issues

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
output_dir = "parsed_synapses"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Initialize variables
output_file = os.path.join(output_dir, 'all_synapses_comprehensive.csv')
total_records = 0
processed_files = 0
error_files = 0
header_written = False

# Get the list of files to process
files_to_process = [f for f in sorted(os.listdir(data_folder)) if f.endswith('.json')]
print(f"Found {len(files_to_process)} JSON schema files")

# Limit the number of files to process
if MAX_FILES_TO_PROCESS > 0:
    files_to_process = files_to_process[:MAX_FILES_TO_PROCESS]

# Create a progress bar
with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
    for file_index, filename in enumerate(files_to_process):
        json_path = os.path.join(data_folder, filename)
        records = []  # Reset records for each file
        file_records = 0

        try:
            # Process the JSON file in chunks
            with open(json_path, 'r') as json_file:
                for line in json_file:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        # Flatten the nested structure
                        flat_record = flatten_dict(record)
                        records.append(flat_record)
                        file_records += 1
                        
                        # Process chunk when it reaches CHUNK_SIZE
                        if len(records) >= CHUNK_SIZE:
                            df_chunk = pd.DataFrame(records)
                            # Append to CSV file
                            df_chunk.to_csv(output_file, mode='a', header=not header_written, index=False)
                            if not header_written:
                                header_written = True
                            
                            total_records += len(records)
                            print(f"Processed {len(records)} records from {filename} (Total synapses: {total_records:,})")
                            
                            # Clear memory immediately
                            del df_chunk
                            records = []
                            gc.collect()
                            
                    except json.JSONDecodeError:
                        continue
                
                # Process remaining records
                if records:
                    df_chunk = pd.DataFrame(records)
                    df_chunk.to_csv(output_file, mode='a', header=not header_written, index=False)
                    if not header_written:
                        header_written = True
                    
                    total_records += len(records)
                    print(f"Processed {len(records)} records from {filename} (Total synapses: {total_records:,})")
                    
                    # Clear memory immediately
                    del df_chunk
                    gc.collect()
                
                if file_records > 0:
                    processed_files += 1
                    print(f"✓ Completed {filename} - Records: {file_records:,} (Total synapses: {total_records:,})")
                else:
                    print(f"No valid records found in {filename}")
                
        except Exception as e:
            error_files += 1
            print(f"Error processing {filename}: {str(e)[:100]}...")
        
        pbar.update(1)

print(f"\nProcessing complete!")
print(f"Total records collected: {total_records:,}")
print(f"Files processed: {processed_files}")
print(f"Errors encountered: {error_files}")

if total_records > 0:
    file_size = os.path.getsize(output_file) / (1024*1024)  # MB
    print(f"\nFINAL SUMMARY:")
    print(f"Total synapses processed: {total_records:,}")
    print(f"Single comprehensive CSV saved to: {os.path.abspath(output_file)}")
    print(f"File size: {file_size:.1f} MB")
    
    # Show a sample of the data structure
    print(f"\nLoading sample to show column structure...")
    try:
        df_sample = pd.read_csv(output_file, nrows=5)
        print(f"Columns in dataset: {list(df_sample.columns)}")
        print(f"Sample data shape: {df_sample.shape}")
        del df_sample  # Clear memory
    except Exception as e:
        print(f"Could not load sample: {e}")
    
    # Force garbage collection
    gc.collect()
    print(f"\nMemory cleanup completed. Final CSV is ready!")
else:
    print("No records were processed!")

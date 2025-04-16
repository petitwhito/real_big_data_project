import pandas as pd
import os
import glob
import re
from datetime import datetime

def extract_info_from_filename(filename):
    """Extract company symbol and timestamp from Boursorama filename."""
    # Get just the filename without path
    base_name = os.path.basename(filename)
    
    # Split by first space to get company symbol
    parts = base_name.split(' ', 1)
    if len(parts) < 2:
        return None, None
        
    symbol = parts[0]
    
    # Extract date and time using regex
    date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', parts[1])
    if date_match:
        timestamp_str = date_match.group(1)
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return symbol, timestamp
        
    return symbol, None

# Base directory for Boursorama data
base_dir = "data/boursorama"

# Maximum number of files to process
MAX_FILES = 3  # Limit to just 3 files

print("Searching for Boursorama data files...")
pickle_files = []

# Look in year directories (if they exist)
for year_dir in glob.glob(os.path.join(base_dir, "20*")):
    if os.path.isdir(year_dir):
        # Look for files that don't end with .tar or .bz2
        files = [f for f in os.listdir(year_dir) 
                if os.path.isfile(os.path.join(year_dir, f)) 
                and not f.endswith(('.tar', '.bz2'))]
        
        if files:
            # Take only a few sample files
            sample_files = files[:MAX_FILES]
            for f in sample_files:
                pickle_files.append(os.path.join(year_dir, f))
            
            # Once we have enough files, break out of the loop
            if len(pickle_files) >= MAX_FILES:
                break

# If no files found in year directories, look directly in the main directory
if not pickle_files:
    files = [f for f in os.listdir(base_dir) 
            if os.path.isfile(os.path.join(base_dir, f)) 
            and not f.endswith(('.tar', '.bz2'))]
    
    if files:
        pickle_files = [os.path.join(base_dir, f) for f in files[:MAX_FILES]]

# Process found files
if pickle_files:
    print(f"Found {len(pickle_files)} files to process (limited to {MAX_FILES})")
    
    for file_path in pickle_files:
        print(f"\nProcessing file: {file_path}")
        
        try:
            # Extract info from filename
            symbol, timestamp = extract_info_from_filename(file_path)
            if symbol and timestamp:
                print(f"Company: {symbol}, Timestamp: {timestamp}")
            
            # Read the pickle file
            df = pd.read_pickle(file_path)
            
            # Display the DataFrame
            print("\nDataFrame Sample:")
            print(df.head(2))  # Only show 2 rows
            print("\nColumns:", df.columns.tolist())
            print("Shape:", df.shape)
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
else:
    print("No suitable Boursorama data files found. Please check the directory structure.")
    
    # List what's actually in the directory
    print("\nDirectory contents:")
    try:
        print(f"Base directory: {base_dir}")
        if os.path.exists(base_dir):
            contents = os.listdir(base_dir)
            print(f"Found {len(contents)} items")
            # Only show first 5 items to avoid excessive output
            for item in contents[:5]:  
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    dir_contents = os.listdir(item_path)
                    print(f"  DIR: {item} ({len(dir_contents)} items)")
                    # Show a few sample filenames from each directory
                    if dir_contents:
                        print(f"    Sample files: {', '.join(dir_contents[:3])}")
                else:
                    print(f"  FILE: {item} ({os.path.getsize(item_path)} bytes)")
        else:
            print("Directory does not exist")
    except Exception as e:
        print(f"Error listing directory: {str(e)}")
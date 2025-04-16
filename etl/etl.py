import pandas as pd
import numpy as np
import sklearn
import glob
import time
import re
import timescaledb_model as tsdb
import mylogging
from datetime import datetime
import os

TSDB = tsdb.TimescaleStockMarketModel
HOME = "/home/bourse/data/"   # we expect subdirectories boursorama and euronext
logger = mylogging.getLogger(__name__, filename="/tmp/bourse.log")

#=================================================
# Extract, Transform and Load data in the database
#=================================================

#
# Helper functions for exploring data
# 

def explore_directory(directory_path):
    """Explore a directory and report its contents."""
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return
    
    logger.info(f"Exploring directory: {directory_path}")
    
    # List contents
    items = os.listdir(directory_path)
    files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
    directories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    
    logger.info(f"Found {len(files)} files and {len(directories)} directories")
    
    # Analyze file types
    file_extensions = {}
    for file in files:
        _, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in file_extensions:
            file_extensions[ext] += 1
        else:
            file_extensions[ext] = 1
    
    logger.info("File types:")
    for ext, count in file_extensions.items():
        logger.info(f"  {ext}: {count} files")
    
    # List directories
    if directories:
        logger.info("Directories:")
        for directory in directories:
            subdir_path = os.path.join(directory_path, directory)
            subdir_items = os.listdir(subdir_path)
            logger.info(f"  {directory}: {len(subdir_items)} items")
    
    return files, directories, file_extensions

def sample_boursorama_file(directory_path, limit=1):
    """Sample and analyze a Boursorama file."""
    logger.info(f"Sampling Boursorama file from {directory_path}")
    
    # Look for year directories
    year_dirs = []
    for item in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, item)) and re.match(r'20\d{2}', item):
            year_dirs.append(item)
    
    if not year_dirs:
        logger.error("No year directories found")
        return None
    
    # Select a year directory
    year_dir = os.path.join(directory_path, year_dirs[0])
    logger.info(f"Using year directory: {year_dir}")
    
    # Find files that are not tar or bz2
    files = []
    for item in os.listdir(year_dir):
        file_path = os.path.join(year_dir, item)
        if os.path.isfile(file_path) and not item.endswith(('.tar', '.bz2')):
            files.append(file_path)
    
    if not files:
        logger.error("No suitable files found")
        return None
    
    # Sample files
    sample_files = files[:limit]
    logger.info(f"Sampling {len(sample_files)} files")
    
    results = []
    for file_path in sample_files:
        try:
            logger.info(f"Reading file: {file_path}")
            
            # Extract info from filename
            filename = os.path.basename(file_path)
            parts = filename.split(' ', 1)
            symbol = parts[0] if len(parts) > 0 else "unknown"
            
            date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', filename)
            timestamp = None
            if date_match:
                timestamp_str = date_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Read the file as a pickle
            df = pd.read_pickle(file_path)
            
            # Analyze the DataFrame
            result = {
                'file_path': file_path,
                'filename': filename,
                'symbol': symbol,
                'timestamp': timestamp,
                'shape': df.shape,
                'columns': list(df.columns),
                'index_name': df.index.name,
                'data_sample': df.head(3)
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    
    return results

def sample_euronext_file(directory_path, limit=1):
    """Sample and analyze an Euronext file."""
    logger.info(f"Sampling Euronext file from {directory_path}")
    
    # Find CSV and Excel files
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    all_files = csv_files + excel_files
    
    if not all_files:
        logger.error("No suitable files found")
        return None
    
    # Sample files
    sample_files = all_files[:limit]
    logger.info(f"Sampling {len(sample_files)} files")
    
    results = []
    for file_path in sample_files:
        try:
            logger.info(f"Reading file: {file_path}")
            
            # Read the file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                continue
            
            # Analyze the DataFrame
            result = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'shape': df.shape,
                'columns': list(df.columns),
                'data_sample': df.head(3)
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    
    return results

#
# ETL Functions
# 

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

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} run in {(end_time - start_time):.2f} seconds.")
        return result

    return wrapper

@timer_decorator
def explore_data_sources(db):
    """Explore and report on data sources."""
    logger.info("Exploring data sources")
    
    # Explore boursorama directory
    boursorama_path = os.path.join(HOME, "boursorama")
    if os.path.exists(boursorama_path):
        logger.info("=== Exploring Boursorama Data ===")
        _, bourso_dirs, _ = explore_directory(boursorama_path)
        
        # Sample a Boursorama file
        bourso_samples = sample_boursorama_file(boursorama_path)
        if bourso_samples:
            for i, sample in enumerate(bourso_samples):
                logger.info(f"Boursorama Sample {i+1}:")
                logger.info(f"  File: {sample['filename']}")
                logger.info(f"  Symbol: {sample['symbol']}")
                logger.info(f"  Timestamp: {sample['timestamp']}")
                logger.info(f"  Shape: {sample['shape']}")
                logger.info(f"  Columns: {sample['columns']}")
                logger.info(f"  Index name: {sample['index_name']}")
                logger.info(f"  Data sample:\n{sample['data_sample']}")
    else:
        logger.error(f"Boursorama directory not found at {boursorama_path}")
    
    # Explore euronext directory
    euronext_path = os.path.join(HOME, "euronext")
    if os.path.exists(euronext_path):
        logger.info("=== Exploring Euronext Data ===")
        _, euro_dirs, euro_ext = explore_directory(euronext_path)
        
        # Sample an Euronext file
        euro_samples = sample_euronext_file(euronext_path)
        if euro_samples:
            for i, sample in enumerate(euro_samples):
                logger.info(f"Euronext Sample {i+1}:")
                logger.info(f"  File: {sample['filename']}")
                logger.info(f"  Shape: {sample['shape']}")
                logger.info(f"  Columns: {sample['columns']}")
                logger.info(f"  Data sample:\n{sample['data_sample']}")
    else:
        logger.error(f"Euronext directory not found at {euronext_path}")
    
    # Check database tables
    logger.info("=== Checking Database Tables ===")
    markets = db.df_query("SELECT * FROM markets")
    logger.info(f"Markets table: {len(markets)} rows")
    
    companies = db.df_query("SELECT * FROM companies")
    logger.info(f"Companies table: {len(companies)} rows")
    
    stocks_count = db.raw_query("SELECT count(*) FROM stocks")
    logger.info(f"Stocks table: {stocks_count[0][0] if stocks_count else 0} rows")
    
    daystocks_count = db.raw_query("SELECT count(*) FROM daystocks")
    logger.info(f"Daystocks table: {daystocks_count[0][0] if daystocks_count else 0} rows")

@timer_decorator
def load_boursorama_file(file_path):
    """Load a single Boursorama file."""
    try:
        # Extract info from filename
        symbol, timestamp = extract_info_from_filename(file_path)
        
        # Read the pickle file
        df = pd.read_pickle(file_path)
        
        # If df is using symbol as index, reset the index
        if df.index.name == 'symbol':
            df = df.reset_index()
        
        # Add timestamp from filename
        if timestamp:
            df['timestamp'] = timestamp
        
        return df, symbol, timestamp
    except Exception as e:
        logger.error(f"Error loading Boursorama file {file_path}: {str(e)}")
        return None, None, None

@timer_decorator
def load_euronext_file(file_path):
    """Load a single Euronext file."""
    try:
        # Read based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        return df
    except Exception as e:
        logger.error(f"Error loading Euronext file {file_path}: {str(e)}")
        return None

# Main ETL function - basic implementation to start with
@timer_decorator
def store_files(start, end, website, db):
    """Process files and store in database."""
    logger.info(f"Processing {website} files from {start} to {end}")
    
    # Convert dates
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Process based on website
    if website.lower() == "boursorama":
        # Process Boursorama data (sample implementation)
        directory_path = os.path.join(HOME, "boursorama")
        
        # Get all year directories
        year_dirs = []
        for item in os.listdir(directory_path):
            dir_path = os.path.join(directory_path, item)
            if os.path.isdir(dir_path) and re.match(r'20\d{2}', item):
                year = int(item)
                # Only process years within our date range
                if year >= start_date.year and year <= end_date.year:
                    year_dirs.append(dir_path)
        
        logger.info(f"Found {len(year_dirs)} year directories to process")
        
        # Process first 5 files from each year as a sample
        sample_limit = 5
        for year_dir in year_dirs:
            year = os.path.basename(year_dir)
            logger.info(f"Sampling files from year {year}")
            
            # Find files (not tar or bz2)
            files = []
            for item in os.listdir(year_dir):
                file_path = os.path.join(year_dir, item)
                if os.path.isfile(file_path) and not item.endswith(('.tar', '.bz2')):
                    files.append(file_path)
            
            sample_files = files[:sample_limit]
            logger.info(f"Processing {len(sample_files)} sample files from {year}")
            
            for file_path in sample_files:
                df, symbol, timestamp = load_boursorama_file(file_path)
                if df is not None:
                    logger.info(f"Loaded file {os.path.basename(file_path)}")
                    logger.info(f"Data shape: {df.shape}")
                
                # Here we would continue with transforming and loading to database
                # For now, just log the data
        
    elif website.lower() == "euronext":
        # Process Euronext data (sample implementation)
        directory_path = os.path.join(HOME, "euronext")
        
        # Find all CSV and Excel files
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
        
        all_files = csv_files + excel_files
        sample_limit = 5
        sample_files = all_files[:sample_limit]
        
        logger.info(f"Processing {len(sample_files)} sample Euronext files")
        
        for file_path in sample_files:
            df = load_euronext_file(file_path)
            if df is not None:
                logger.info(f"Loaded file {os.path.basename(file_path)}")
                logger.info(f"Data shape: {df.shape}")
            
            # Here we would continue with transforming and loading to database
            # For now, just log the data
    
    else:
        logger.error(f"Unknown website: {website}")
    
    return "Sample processing complete"

if __name__ == '__main__':
    print("Starting ETL process")
    pd.set_option('display.max_columns', None)  # useful for debugging
    
    # Connect to database
    db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'db', 'monmdp')  # inside docker
    
    # First, explore the data to understand its structure
    explore_data_sources(db)
    
    # Then process sample files to test basic functionality
    store_files("2020-01-01", "2020-02-01", "euronext", db)  # one month to test
    store_files("2020-01-01", "2020-02-01", "boursorama", db)  # one month to test
    
    print("ETL process completed")
    
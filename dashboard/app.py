# Force unbuffered output
import sys
print("===== DASHBOARD STARTUP =====")

# Core imports
import numpy as np
import pandas as pd
import datetime
import dash
import dash_bootstrap_components as dbc

sys.stdout.reconfigure(line_buffering=True)


# Try to connect to the database
print("Attempting database connection...")
try:
    import timescaledb_model as tsdb
    db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'db', 'monmdp')
    print("Database connection successful!")
    
    # Basic database test
    try:
        companies_count = db.df_query("SELECT COUNT(*) FROM companies").iloc[0, 0]
        print(f"Found {companies_count} companies in database")
    except Exception as e:
        print(f"Error querying companies: {str(e)}")
except Exception as e:
    print(f"Database connection failed: {str(e)}")
    # Create a dummy database object for testing
    db = None

# Initialize Dash app
print("Initializing Dash application...")
external_stylesheets=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, title="Bourse", suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets, assets_ignore='style.css?v=1.0')
app.df = pd.DataFrame()
app.daydf = pd.DataFrame()
app.comp_names = []
server = app.server

# Import layout and tabs
from index import layout  # Not before app is defined since we use it
app.layout = layout

print("===== DASHBOARD INITIALIZED =====")

if __name__ == '__main__':
    app.run(debug=True)
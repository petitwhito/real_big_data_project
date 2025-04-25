import pandas as pd
import numpy as np
import re
import timescaledb_model as tsdb
import mylogging
from datetime import datetime
import os
import sys
import traceback
import time
import bz2  # Ajout pour gérer les fichiers .bz2
# from io import BytesIO # No longer needed for bz2 reading

# Configuration de base
HOME = "/home/bourse/data/"  # Répertoires boursorama et euronext attendus
logger = mylogging.getLogger(__name__, filename="/tmp/bourse.log")

# Forcer le vidage immédiat de la sortie pour les environnements Docker
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Statistiques et suivi du traitement
time_stats = {}
processed_files = set()

# Structure des statistiques de normalisation (Optional, can be removed if not used for logging)
symbol_normalization_stats = {
    'prefix_removed': 0,
    'special_cases': 0,
    'symbols_processed': 0
}

#=================================================
# SECTION 1: FONCTIONS UTILITAIRES ET LOGGING
#=================================================

def log_info(message):
    """Log une information et l'affiche sur la sortie standard"""
    print(message, flush=True)
    logger.info(message)

def log_error(message):
    """Log une erreur et l'affiche sur la sortie standard"""
    print(f"ERROR: {message}", flush=True)
    logger.error(message)

def timer_decorator(func):
    """Mesure le temps d'exécution d'une fonction"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        log_info(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        log_info(f"{func.__name__} completed in {duration:.2f} seconds.")
        time_stats[func.__name__] = duration
        return result
    return wrapper

#=================================================
# SECTION 2: FONCTIONS DE TRAITEMENT DES SYMBOLES
#=================================================

# Définition des mappages de marché centralisés
MARKET_PREFIXES = {
    '1rP': ('paris', 6),      # Paris
    '1rA': ('amsterdam', 5),  # Amsterdam
    '1u': ('lse', 2),         # London
    '1g': ('milano', 3),      # Milan
    'FF55-': ('madrid', 4),   # Mercados Espanoles
    '1z': ('xetra', 7),       # Deutsche Borse (Xetra)
    'FF11_': ('bruxelle', 8), # Bruxelle
    '1b': ('bruxelle', 8)     # Bruxelle - autre préfixe
}

# Cas spéciaux pour certains symboles
SPECIAL_CASES = {
    '1rPAALMIL': ('ALMIL', 6),  # 1000MERCIS
    '1rPAGECP': ('GECP', 6)     # GECI INTL
}

# REMOVED: clean_company_name function (replaced by vectorized operation)
# def clean_company_name(name):
#    ...

# --- MODIFIED: normalize_symbol_and_market TO USE CACHE ---
def normalize_symbol_and_market(symbol, mid=6, cache=None): # Add cache parameter
    """
    Normalise un symbole Boursorama en supprimant le préfixe
    et détermine l'ID du marché correct (avec cache)

    Args:
        symbol: Chaîne de symbole originale
        mid: ID du marché original (défaut: 6 pour Paris)
        cache: Dictionary to use for caching results

    Returns:
        tuple: (symbole_normalisé, id_marché)
    """
    if not symbol or not isinstance(symbol, str):
        return symbol, mid

    # Use cache if provided
    if cache is not None:
        cache_key = (symbol, mid)
        if cache_key in cache:
            return cache[cache_key]

    # Mettre à jour le compteur global (optional)
    # symbol_normalization_stats['symbols_processed'] += 1

    result = None
    # Vérifier d'abord les cas spéciaux
    if symbol in SPECIAL_CASES:
        normalized, market_id = SPECIAL_CASES[symbol]
        # symbol_normalization_stats['special_cases'] += 1 # Optional
        result = (normalized, market_id)

    # Vérifier le préfixe et le supprimer si trouvé
    if result is None:
        for prefix, (_, market_id) in MARKET_PREFIXES.items():
            if symbol.startswith(prefix):
                # symbol_normalization_stats['prefix_removed'] += 1 # Optional
                normalized_symbol = symbol[len(prefix):]
                result = (normalized_symbol, market_id)
                break # Found prefix, stop searching

    # Si aucun préfixe trouvé, renvoyer l'original
    if result is None:
        result = (symbol, mid)

    # Store in cache if provided
    if cache is not None:
        cache[cache_key] = result

    return result
# --- END CACHE MODIFICATION ---

#=================================================
# SECTION 3: CLASSE DE TRAITEMENT PRINCIPAL
#=================================================

class Processor:
    """Processeur de données boursières avec traitement sophistiqué"""

    def __init__(self, db_model):
        """Initialise le processeur avec le modèle de base de données"""
        self.db = db_model
        self.companies_save = None  # DataFrame pour stocker toutes les entreprises
        self.companies_batch = []   # Lot d'entreprises à enregistrer
        self.stocks_batch = []      # Lot d'actions à enregistrer
        self.day_batch = []         # Lot d'actions pour agrégation quotidienne
        self.daystocks_batch = []   # Lot d'actions agrégées quotidiennement
        self.market_cache = {}      # Cache pour les IDs de marché
        # OPTIMIZATION: Initialize symbol cache within the class instance
        self.symbol_normalization_cache = {}
        self.isin_to_id_map = {}    # Map des ISINs vers IDs d'entreprises

        # Charger le cache de marché
        self._load_market_cache()

    def _load_market_cache(self):
        """Charge les IDs de marché depuis la base de données"""
        markets_df = self.db.df_query("SELECT id, alias FROM markets")
        self.market_cache = dict(zip(markets_df['alias'], markets_df['id'])) if not markets_df.empty else {}
        log_info(f"Market cache loaded with {len(self.market_cache)} markets")

    def get_market_id(self, market_alias):
        """Obtient l'ID de marché depuis l'alias, avec repli vers Paris"""
        return self.market_cache.get(market_alias, 6)  # Par défaut Paris (ID 6)

    def process_boursorama_file(self, file_path):
        """Traite un fichier Boursorama individuel (normal ou .bz2)"""
        if file_path in processed_files:
            return 0, 0

        try:
            filename = os.path.basename(file_path)

            # Extraire l'alias du marché et la date du nom de fichier
            if filename.endswith('.bz2'):
                base_filename = filename[:-4]
                parts = base_filename.split(' ', 1)
            else:
                parts = filename.split(' ', 1)

            if len(parts) < 2:
                # log_error(f"Invalid filename format: {filename}") # Keep logging minimal if needed
                return 0, 0

            alias = parts[0]
            date_str = parts[1]

            # Traiter les dates avec underscores (_)
            try:
                if '_' in date_str:
                    date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
                    hour_part = date_str.replace(date_part, '').strip()
                    if hour_part.startswith(' '):
                        hour_part = hour_part[1:]
                    hour_part_fixed = hour_part.replace('_', ':')
                    formatted_date_str = f"{date_part} {hour_part_fixed}"
                    timestamp = pd.to_datetime(formatted_date_str)
                else:
                    timestamp = pd.to_datetime(date_str)

            except Exception as e:
                # log_error(f"Error parsing date in {filename}: {str(e)}")
                day_part = extract_boursorama_day(date_str)
                if day_part:
                    timestamp = pd.to_datetime(day_part)
                else:
                    # raise ValueError(f"Unable to parse date from filename: {filename}") # Avoid exceptions
                    return 0, 0

            # --- OPTIMIZED BZ2 READING ---
            df = None
            try:
                if filename.endswith('.bz2'):
                    with bz2.open(file_path, 'rb') as f:
                        df = pd.read_pickle(f) # Read directly from file object
                else:
                    df = pd.read_pickle(file_path)
            except Exception as read_error:
                log_error(f"Error reading file {file_path}: {str(read_error)}")
                return 0, 0
            # --- END OPTIMIZED BZ2 READING ---

            if df is None or df.empty:
                return 0, 0

            # Gérer le cas où le symbole est à la fois index et colonne
            if df.index.name == 'symbol' and 'symbol' in df.columns:
                df = df.drop(columns=['symbol'])

            if df.index.name == 'symbol':
                df = df.reset_index()

            if 'symbol' not in df.columns:
                # log_error(f"No symbol column in {filename}")
                return 0, 0

            mid = self.get_market_id(alias)
            df["mid"] = mid
            df["date"] = timestamp

            # --- VECTORIZED NAME CLEANING ---
            if 'name' in df.columns:
                # Ensure 'name' is string type before using .str accessor
                df['name'] = df['name'].astype(str)
                # Use str.removeprefix if pandas version >= 1.4, otherwise use slicing
                try:
                    df['name'] = df['name'].str.removeprefix('SRD')
                except AttributeError: # Handle older pandas versions
                    srd_mask = df['name'].str.startswith('SRD', na=False)
                    df.loc[srd_mask, 'name'] = df.loc[srd_mask, 'name'].str[3:]
            # --- END VECTORIZED NAME CLEANING ---

            if 'name' not in df.columns:
                df['name'] = df['symbol']

            # Traiter la colonne 'last'
            if 'last' in df.columns:
                if df['last'].dtype == 'object':
                    df['last'] = df['last'].str.replace(r'\([a-zA-Z]\)|\s+', '', regex=True)
                    df['last'] = df['last'].str.replace(',', '.', regex=False)
                df['last'] = pd.to_numeric(df['last'], errors='coerce')
            else:
                if 'value' in df.columns:
                    df['last'] = pd.to_numeric(df['value'], errors='coerce')
                elif 'close' in df.columns:
                    df['last'] = pd.to_numeric(df['close'], errors='coerce')
                else:
                    # log_error(f"No price column (last/value/close) in {filename}")
                    return 0, 0

            # Ajouter le volume s'il manque
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                if df['volume'].dtype == 'object':
                    df['volume'] = df['volume'].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

            # Filtrer les volumes nuls et valeurs invalides (CORE LOGIC - UNCHANGED)
            valid_mask = (df['last'] > 0) & (df['volume'] > 0)
            df = df[valid_mask].dropna(subset=['last'])

            if df.empty:
                return 0, 0

            # Traiter le DataFrame
            companies_added = self.process_dataframe(df)

            processed_files.add(file_path)
            return companies_added, len(df)

        except Exception as e:
            log_error(f"Error processing Boursorama file {file_path}: {str(e)}")
            return 0, 0

    def process_dataframe(self, df):
        """
        Traite le DataFrame d'un fichier en extrayant ses données et en les mettant
        dans les tables d'entreprise et d'actions.
        """
        # Traiter les entreprises
        companies_df = self.__process_companies(df)
        companies_added = 0
        if companies_df is not None and not companies_df.empty:
             companies_added = len(companies_df)
             self.companies_batch.append(companies_df)


        # Traiter les actions
        stocks_df = self.__process_stocks(df)

        # Ajouter aux lots
        if stocks_df is not None and not stocks_df.empty:
            self.stocks_batch.append(stocks_df)
            self.day_batch.append(stocks_df)

        return companies_added

    def __process_companies(self, df):
        """Traite le DataFrame et remplit la table des entreprises - Version ultra-optimisée"""
        if not {'symbol', 'name'}.issubset(df.columns):
            return pd.DataFrame()

        companies = df[['symbol', 'name', 'mid']].drop_duplicates('symbol').dropna(subset=['symbol'])

        has_isin = 'isin' in df.columns
        if has_isin:
            companies['isin'] = df['isin'].copy()

        if companies.empty:
            return pd.DataFrame()

        symbols = companies['symbol'].values
        mids = companies['mid'].values

        normalized_symbols = []
        normalized_mids = []

        # --- USE SYMBOL CACHE ---
        for i, (symbol, mid) in enumerate(zip(symbols, mids)):
            # Pass self.symbol_normalization_cache to the function
            norm_symbol, norm_mid = normalize_symbol_and_market(symbol, mid, self.symbol_normalization_cache)
            normalized_symbols.append(norm_symbol)
            normalized_mids.append(norm_mid)
        # --- END USE SYMBOL CACHE ---

        companies['symbol'] = normalized_symbols
        companies['mid'] = normalized_mids

        companies = companies.drop_duplicates('symbol')
        companies.set_index('symbol', inplace=True)

        if self.companies_save is None:
            existing_companies = self.db.df_query("SELECT id, name, symbol, mid, isin FROM companies")
            if not existing_companies.empty:
                existing_companies.set_index('symbol', inplace=True)
                self.companies_save = existing_companies
                if 'isin' in existing_companies.columns:
                    isin_map = existing_companies[~existing_companies['isin'].isna()][['id', 'isin']]
                    self.isin_to_id_map = dict(zip(isin_map['isin'], isin_map['id']))
            else:
                self.companies_save = pd.DataFrame(columns=['id', 'name', 'mid', 'isin'])
                self.companies_save.index.name = 'symbol'

        # ISIN update logic (remains the same)
        if has_isin and self.isin_to_id_map:
            for idx, row in companies.iterrows():
                 if 'isin' in row and pd.notna(row['isin']) and row['isin'] in self.isin_to_id_map:
                    existing_id = self.isin_to_id_map[row['isin']]
                    if idx not in self.companies_save.index:
                        company_data = {'id': existing_id, 'name': row['name'], 'mid': row['mid'], 'isin': row['isin']}
                        self.companies_save.loc[idx] = company_data
                        update_query = "UPDATE companies SET name = %s, symbol = %s WHERE id = %s"
                        self.db.execute(update_query, [row['name'], idx, existing_id])


        existing_symbols = set(self.companies_save.index)
        current_symbols = set(companies.index)
        new_symbols = current_symbols - existing_symbols

        if not new_symbols:
            return pd.DataFrame()

        new_companies = companies.loc[list(new_symbols)].copy()

        if not new_companies.empty:
            next_id = 1
            # Handle case where companies_save might be empty or max() returns NaN
            if not self.companies_save.empty and 'id' in self.companies_save.columns:
                 max_id = self.companies_save['id'].max()
                 if pd.notna(max_id):
                     next_id = min(32700, int(max_id) + 1)


            if next_id + len(new_companies) > 32767:
                log_error(f"Company ID limit reached (32767). Skipping {len(new_companies) - (32767 - next_id)} companies.")
                new_companies = new_companies.iloc[:(32767 - next_id)]
                if new_companies.empty:
                    return pd.DataFrame()

            new_companies['id'] = np.arange(next_id, next_id + len(new_companies), dtype=np.int16)

            if has_isin:
                for idx, row in new_companies.iterrows():
                    if 'isin' in row and pd.notna(row['isin']):
                        self.isin_to_id_map[row['isin']] = row['id']

            # Use concat instead of append for modern pandas
            self.companies_save = pd.concat([self.companies_save, new_companies])


        return new_companies

    def __process_stocks(self, df):
        """Traite le DataFrame et remplit la table des actions - Version ultra-optimisée"""
        if self.companies_save is None or self.companies_save.empty:
            return pd.DataFrame()

        # Determine required price column
        required_cols = {'symbol', 'date'}
        price_col = None
        if 'last' in df.columns:
            price_col = 'last'
        elif 'value' in df.columns:
            price_col = 'value'
        elif 'close' in df.columns:
            price_col = 'close'
        else:
            return pd.DataFrame() # No price column found
        required_cols.add(price_col)

        if not required_cols.issubset(df.columns):
             return pd.DataFrame()


        # --- OPTIMIZED SYMBOL/CID MAPPING ---
        # 1. Select necessary columns including the ORIGINAL symbol and price column
        cols_to_keep = ['symbol', price_col, 'date']
        if 'volume' in df.columns:
            cols_to_keep.append('volume')
        stocks_subset = df[cols_to_keep].copy()

        # Rename price column to 'value' for consistency
        stocks_subset.rename(columns={price_col: 'value'}, inplace=True)

        # 2. Add volume if missing, filter zero volume (CORE LOGIC)
        if 'volume' not in stocks_subset.columns:
            stocks_subset['volume'] = 0
        # Ensure volume is numeric before filtering
        stocks_subset['volume'] = pd.to_numeric(stocks_subset['volume'], errors='coerce').fillna(0)
        stocks_subset = stocks_subset[stocks_subset['volume'] > 0]


        if stocks_subset.empty:
            return pd.DataFrame()

        # 3. Re-normalize symbols in this subset (use cache)
        normalized_symbols = []
        # Assume default mid=6 if not present in df, otherwise get it
        default_mid = 6
        mids_to_use = df['mid'] if 'mid' in df.columns else [default_mid] * len(stocks_subset)

        for symbol, mid in zip(stocks_subset['symbol'], mids_to_use):
             norm_symbol, _ = normalize_symbol_and_market(symbol, mid, self.symbol_normalization_cache)
             normalized_symbols.append(norm_symbol)
        stocks_subset['normalized_symbol'] = normalized_symbols


        # 4. Use pd.Series.map for efficient mapping from NORMALIZED symbol to CID
        # Ensure companies_save index is 'symbol' (normalized)
        if self.companies_save.index.name != 'symbol':
             if 'symbol' in self.companies_save.columns:
                 self.companies_save = self.companies_save.set_index('symbol')
             else:
                 log_error("companies_save DataFrame is missing 'symbol' index/column during stock processing.")
                 return pd.DataFrame() # Cannot proceed

        symbol_to_cid_map = self.companies_save['id']
        stocks_subset['cid'] = stocks_subset['normalized_symbol'].map(symbol_to_cid_map)


        # 5. Filter rows where mapping failed (no corresponding company found)
        stocks_subset = stocks_subset.dropna(subset=['cid'])

        if stocks_subset.empty:
            return pd.DataFrame()

        # 6. Create final stocks DataFrame with correct types
        stocks_df = pd.DataFrame({
            'cid': stocks_subset['cid'].astype(np.int16),
            'date': stocks_subset['date'],
            'value': pd.to_numeric(stocks_subset['value'], errors='coerce'), # Already renamed
            'volume': stocks_subset['volume'].astype(np.int32) # Already numeric
        })

        # 7. Filter invalid values (CORE LOGIC)
        stocks_df = stocks_df[stocks_df['value'] > 0].dropna(subset=['value'])

        # Set index
        stocks_df = stocks_df.set_index('cid')
        # --- END OPTIMIZED SYMBOL/CID MAPPING ---

        return stocks_df

    def process_daystocks(self, date):
        """
        À partir du lot d'actions traité, crée des actions quotidiennes
        """
        if not self.day_batch:
            return

        # Concaténer toutes les actions de ce jour
        try:
            # Use copy=False if pandas version allows and memory is tight
            daystocks = pd.concat(self.day_batch) # copy=False removed for broader compatibility
        except Exception as concat_error:
             log_error(f"Error concatenating day_batch for date {date}: {concat_error}")
             self.day_batch = [] # Clear batch on error
             return


        if daystocks.empty:
            self.day_batch = []  # Réinitialiser le lot journalier
            return

        # Ensure index is 'cid' before grouping
        if daystocks.index.name != 'cid':
             if 'cid' in daystocks.columns:
                 daystocks = daystocks.reset_index().set_index('cid') # Reset and set index
             else:
                 log_error(f"Missing 'cid' for grouping daystocks on {date}")
                 self.day_batch = []
                 return


        # Optimisation: Utiliser agg avec NamedAgg pour une agrégation efficace
        try:
            aggregated_daystocks = daystocks.groupby(level='cid').agg( # Group by index 'cid'
                open=pd.NamedAgg(column='value', aggfunc='first'),
                close=pd.NamedAgg(column='value', aggfunc='last'),
                high=pd.NamedAgg(column='value', aggfunc='max'),
                low=pd.NamedAgg(column='value', aggfunc='min'),
                volume=pd.NamedAgg(column='volume', aggfunc='sum')
            )
        except Exception as agg_error:
             log_error(f"Error aggregating daystocks for date {date}: {agg_error}")
             self.day_batch = [] # Clear batch on error
             return


        # Définir la date
        aggregated_daystocks['date'] = date

        # Ajouter des statistiques supplémentaires
        aggregated_daystocks['mean'] = (aggregated_daystocks['open'] + aggregated_daystocks['close'] + aggregated_daystocks['high'] + aggregated_daystocks['low']) / 4

        # Ajouter au lot daystocks
        self.daystocks_batch.append(aggregated_daystocks)

        # Nettoyer le lot journalier
        self.day_batch = []

    def clean_stocks(self):
        """Nettoyage optimisé des données de stocks"""
        if not self.stocks_batch:
            return

        try:
            # Combiner tous les lots
            stocks = pd.concat(self.stocks_batch) # copy=False removed
        except Exception as concat_error:
            log_error(f"Error concatenating stocks_batch for cleaning: {concat_error}")
            self.stocks_batch = [] # Clear batch on error
            return


        if stocks.empty:
            self.stocks_batch = []
            return

        # Reset index et tri
        stocks.reset_index(inplace=True)

        # Optimisation 1: Utiliser sort_values avec inplace
        # mergesort is stable, good if original order matters within duplicates
        stocks.sort_values(['cid', 'date'], inplace=True, kind='mergesort')

        # Optimisation 2: Conversion plus efficace du jour
        stocks['day'] = stocks['date'].dt.floor('D')

        # Calculer les différences entre valeurs consécutives
        # Use groupby().transform('shift') for potentially better performance on large groups
        stocks['value_prev'] = stocks.groupby('cid')['value'].shift(1)
        stocks['value_change'] = (stocks['value'] - stocks['value_prev']).abs()
        # Avoid division by zero or near-zero
        stocks['pct_change'] = (stocks['value_change'] / stocks['value_prev'].abs().replace(0, np.nan))


        # Garder les lignes avec changement significatif (plus de 0.1%)
        min_change_pct = 0.001  # 0.1%
        # Handle NaN pct_change (e.g., first entry or zero prev value)
        has_change = stocks['pct_change'].fillna(0) > min_change_pct


        # Ajouter les première et dernière valeurs de chaque jour
        # Use keep='first' (default) and keep='last'
        first_of_day = ~stocks.duplicated(['cid', 'day'])
        last_of_day = ~stocks.duplicated(['cid', 'day'], keep='last')

        # Ajouter les première et dernière valeurs pour chaque ID d'entreprise
        first_of_cid = ~stocks.duplicated(['cid'])
        last_by_cid = ~stocks.duplicated(['cid'], keep='last')

        # Combiner les critères
        keep_mask = has_change | first_of_day | last_of_day | first_of_cid | last_by_cid

        # Appliquer le filtre et nettoyer
        stocks = stocks.loc[keep_mask, ['cid', 'date', 'value', 'volume']].copy() # Select columns explicitly
        # stocks.drop(columns=['value_prev', 'value_change', 'pct_change', 'day'], inplace=True) # Already dropped by selection
        stocks.set_index('cid', inplace=True)

        # Remplacer le lot
        self.stocks_batch = [stocks] if not stocks.empty else []


    def commit_companies(self):
        """Enregistrer toutes les entreprises dans la base de données"""
        if not self.companies_batch:
            return 0

        total_committed = 0

        try:
            # Combine all batches first
            all_new_companies = pd.concat(self.companies_batch)
            if all_new_companies.empty:
                 self.companies_batch = []
                 return 0

            # Obtenir les IDs d'entreprise existants pour éviter les doublons
            existing_ids_df = self.db.df_query("SELECT id FROM companies")
            existing_ids = set(existing_ids_df['id'].tolist()) if not existing_ids_df.empty else set()

            # Préparer pour l'insertion dans la base de données
            companies_to_insert_df = all_new_companies.reset_index()

            # Filtrer les entreprises qui n'existent pas déjà
            mask = ~companies_to_insert_df['id'].isin(existing_ids)
            final_new_companies = companies_to_insert_df[mask]

            if not final_new_companies.empty:
                # Utiliser la méthode du modèle DB pour l'insertion efficace
                self.db.df_write(
                    final_new_companies,
                    'companies',
                    commit=True,
                    if_exists="append",
                    index=False
                )
                total_committed = len(final_new_companies)
                # No need to update existing_ids set here as we commit once

        except Exception as e:
            log_error(f"Error committing companies: {str(e)}")
            # Rollback might be handled within db.df_write or needs explicit call
            if hasattr(self.db, 'connection') and self.db.connection:
                 try:
                     self.db.connection.rollback()
                 except Exception as rb_err:
                     log_error(f"Rollback failed: {rb_err}")
            total_committed = 0 # Indicate failure

        finally:
            # Réinitialiser le lot d'entreprises regardless of success/failure
            self.companies_batch = []

        return total_committed


    def commit_stocks(self):
        """Enregistrer toutes les actions dans la base de données"""
        if not self.stocks_batch:
            return 0

        total_committed = 0
        try:
            # Combine all batches first
            all_stocks = pd.concat(self.stocks_batch)
            if all_stocks.empty:
                 self.stocks_batch = []
                 return 0

            # Vérifier que l'index est bien 'cid'
            if all_stocks.index.name != 'cid':
                if 'cid' in all_stocks.columns:
                    all_stocks = all_stocks.reset_index().set_index('cid')
                else:
                    log_error("Cannot commit stocks: 'cid' index or column missing.")
                    self.stocks_batch = []
                    return 0

            # Vérifier que le DataFrame n'est pas vide après le réindexage
            if all_stocks.empty:
                 self.stocks_batch = []
                 return 0

            # CRUCIAL: Appeler df_write avec index=True et index_label='cid'
            self.db.df_write(
                all_stocks,
                'stocks',
                commit=True,
                index=True,
                index_label='cid'
            )
            total_committed = len(all_stocks)

        except Exception as e:
            log_error(f"Error committing stocks: {str(e)}")
            if hasattr(self.db, 'connection') and self.db.connection:
                 try:
                     self.db.connection.rollback()
                 except Exception as rb_err:
                     log_error(f"Rollback failed: {rb_err}")
            total_committed = 0
        finally:
            # Réinitialiser le lot d'actions
            self.stocks_batch = []

        return total_committed


    def commit_daystocks(self):
        """Enregistrer toutes les actions quotidiennes dans la base de données"""
        if not self.daystocks_batch:
            return 0

        total_committed = 0
        try:
            # Combine all batches
            all_daystocks = pd.concat(self.daystocks_batch)
            if all_daystocks.empty:
                 self.daystocks_batch = []
                 return 0

            # Ensure index is 'cid'
            if all_daystocks.index.name != 'cid':
                 if 'cid' in all_daystocks.columns:
                     all_daystocks = all_daystocks.reset_index().set_index('cid')
                 else:
                     log_error("Cannot commit daystocks: 'cid' index or column missing.")
                     self.daystocks_batch = []
                     return 0

            if all_daystocks.empty:
                 self.daystocks_batch = []
                 return 0

            # Écrire dans la base de données
            self.db.df_write(all_daystocks, 'daystocks', commit=True, index=True, index_label='cid')
            total_committed = len(all_daystocks)

        except Exception as e:
            log_error(f"Error committing daystocks: {str(e)}")
            if hasattr(self.db, 'connection') and self.db.connection:
                 try:
                     self.db.connection.rollback()
                 except Exception as rb_err:
                     log_error(f"Rollback failed: {rb_err}")
            total_committed = 0
        finally:
            # Réinitialiser le lot d'actions quotidiennes
            self.daystocks_batch = []

        return total_committed


    def commit_all(self):
        """Enregistrer toutes les données en attente dans la base de données"""
        companies = self.commit_companies()
        stocks = self.commit_stocks()
        daystocks = self.commit_daystocks()
        return companies, stocks, daystocks

    def update_company_info(self, symbol, new_name, old_name, isin=None):
        """Met à jour le nom et le symbole d'une entreprise si l'ISIN est identique"""
        # This function seems less critical for the main ETL flow and might be complex to integrate
        # with the batch processing. Consider if it's essential or can be handled separately.
        if isin:
            # Vérifier si l'ISIN existe déjà avec un nom/symbole différent
            query = "SELECT id, name, symbol FROM companies WHERE isin = %s"
            result = self.db.df_query(query, [isin])

            if not result.empty:
                company_id = result['id'].iloc[0]
                # Mettre à jour l'entreprise
                update_query = "UPDATE companies SET name = %s, symbol = %s WHERE id = %s"
                self.db.execute(update_query, [new_name, symbol, company_id])
                log_info(f"Updated company: {old_name} → {new_name} (ID: {company_id})")
                return True

        return False

#=================================================
# SECTION 4: FONCTIONS DE TRAITEMENT DE FICHIERS
#=================================================

# --- MODIFIED: load_euronext_file ---
def load_euronext_file(file_path):
    """Charge et analyse un fichier Euronext (CSV ou Excel)"""
    try:
        df = None
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, delimiter='\t', skiprows=None,
                            on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return None

        if df is None or df.empty:
             return None

        # Standardiser les noms de colonnes
        column_mapping = {
            'Symbol': 'symbol', 'Name': 'name', 'Last': 'last',
            'Volume': 'volume', 'ISIN': 'isin'
        }
        # Clean potential whitespace and rename efficiently
        df.columns = df.columns.str.strip()
        rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        df.rename(columns=rename_dict, inplace=True)


        if 'symbol' not in df.columns:
            return None

        # VECTORIZED NAME CLEANING
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str)
            try:
                df['name'] = df['name'].str.removeprefix('SRD')
            except AttributeError: # Handle older pandas versions
                srd_mask = df['name'].str.startswith('SRD', na=False)
                df.loc[srd_mask, 'name'] = df.loc[srd_mask, 'name'].str[3:]
        # END VECTORIZED NAME CLEANING

        # Convert 'last' and 'volume' efficiently
        if 'last' in df.columns:
             df['last'] = pd.to_numeric(df['last'].astype(str).str.replace(r'[^\d.,]+', '', regex=True).str.replace(',', '.', regex=False), errors='coerce')
        if 'volume' not in df.columns:
             df['volume'] = 0
        elif 'volume' in df.columns:
             df['volume'] = pd.to_numeric(df['volume'].astype(str).str.replace(r'[^\d]+', '', regex=True), errors='coerce').fillna(0)


        # Filter invalid data (CORE LOGIC)
        valid_mask = (df['volume'] > 0) & (df['last'] > 0)
        df = df[valid_mask].dropna(subset=['last'])


        if df.empty:
            return None

        # Extraire la date du nom de fichier
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            file_date = date_match.group(1)
            df['date'] = pd.to_datetime(file_date)
            df['mid'] = 6 # Default Paris

            # Extract market ID
            market_match = re.search(r'_([A-Za-z]+)_', filename)
            if market_match:
                market_name = market_match.group(1).lower()
                market_map = {'amsterdam': 5, 'london': 2, 'milan': 3, 'madrid': 4, 'brussels': 8, 'xetra': 7}
                if market_name in market_map:
                    df['mid'] = market_map[market_name]
                    # log_info(f"{market_name.capitalize()} market detected in {filename}") # Optional

            return df
        else:
            log_error(f"Could not extract date from Euronext filename: {filename}")
            return None

    except Exception as e:
        log_error(f"Error loading Euronext file {file_path}: {str(e)}")
        return None
# --- END MODIFICATION ---

def resolve_conflicting_values(boursorama_value, euronext_value):
    """Décide quelle valeur utiliser en cas de conflit"""
    # This function is not currently used in the main ETL flow.
    # If needed, ensure it handles None values correctly.
    if boursorama_value is not None and euronext_value is not None:
        # Example strategy: prioritize Euronext if difference is large
        if abs(boursorama_value - euronext_value) > 0.1 * abs(euronext_value): # Use abs for comparison
            return euronext_value
        return (boursorama_value + euronext_value) / 2
    return boursorama_value if boursorama_value is not None else euronext_value

def extract_boursorama_day(date_str):
    """
    Extrait la partie date (YYYY-MM-DD) d'une chaîne de date Boursorama.
    """
    try:
        # Match YYYY-MM-DD at the beginning of the relevant part
        match = re.match(r'(\d{4}-\d{2}-\d{2})', date_str)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None

@timer_decorator
def process_boursorama_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Boursorama - Version optimisée mono-thread"""
    log_info(f"Processing Boursorama files from {start_date.date()} to {end_date.date()}")

    processor = Processor(db_model)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    boursorama_path = os.path.join(HOME, "boursorama")

    years_to_process = range(max(start_date.year, 2019), min(end_date.year, 2024) + 1)
    years_to_process = [year for year in years_to_process
                        if os.path.exists(os.path.join(boursorama_path, str(year)))
                        and os.path.isdir(os.path.join(boursorama_path, str(year)))]
    log_info(f"Found Boursorama data for years: {years_to_process}")

    month_filters = {}
    for year in years_to_process:
        start_month = start_date.month if year == start_date.year else 1
        end_month = end_date.month if year == end_date.year else 12
        month_filters[year] = list(range(start_month, end_month + 1))

    files_processed = 0
    companies_added_total = 0
    stocks_added_total = 0 # Track total stocks before commit reset
    daystocks_added_total = 0
    prev_date = None
    date_cache = {}
    processed_files_cache = set(processed_files) # Use global set

    commit_threshold = 50000 # Keep commit threshold reasonable

    for year in years_to_process:
        year_dir = os.path.join(boursorama_path, str(year))
        log_info(f"Scanning files in year directory {year}")
        year_files = []

        # Use os.scandir for potentially faster listing
        try:
            with os.scandir(year_dir) as entries:
                for entry in entries:
                    if not entry.is_file() or not (entry.name.endswith('.pkl') or entry.name.endswith('.bz2')):
                        continue

                    file_path = entry.path
                    if file_path in processed_files_cache:
                        continue

                    try:
                        filename = entry.name
                        base_name = filename[:-4] if filename.endswith('.bz2') else filename
                        parts = base_name.split(' ', 1)
                        if len(parts) < 2: continue

                        date_str = parts[1]
                        day_part = extract_boursorama_day(date_str)
                        if not day_part: continue

                        # Filter by month using string manipulation (faster than datetime conversion)
                        month_str = day_part.split('-')[1]
                        try:
                            month = int(month_str)
                            if month not in month_filters.get(year, []):
                                continue
                        except (ValueError, IndexError):
                            continue

                        # Use date cache for full timestamp conversion
                        if day_part in date_cache:
                            timestamp = date_cache[day_part]
                        else:
                            timestamp = pd.to_datetime(day_part)
                            date_cache[day_part] = timestamp

                        # Final date range check
                        if start_date <= timestamp <= end_date:
                            year_files.append((timestamp, file_path, parts[0])) # (timestamp, path, market_alias)

                    except Exception as parse_err:
                        log_error(f"Error parsing filename {entry.name}: {parse_err}")
                        continue
        except FileNotFoundError:
             log_error(f"Directory not found: {year_dir}")
             continue


        year_files.sort() # Sort by timestamp

        if year_files:
            months_count = {}
            for timestamp, _, _ in year_files:
                month = timestamp.month
                months_count[month] = months_count.get(month, 0) + 1
            log_info(f"Found {len(year_files)} files to process in year {year}")
            log_info(f"Files distribution by month in {year}: {months_count}")

            last_month_logged = None
            files_processed_in_month = 0
            stocks_in_current_commit_batch = 0 # Track stocks for commit threshold

            for timestamp, file_path, market_name in year_files:
                current_month = timestamp.month
                if last_month_logged != current_month:
                    if last_month_logged is not None:
                        log_info(f">>> Completed processing {files_processed_in_month} files for {year}-{last_month_logged:02d}")
                    last_month_logged = current_month
                    files_processed_in_month = 0
                    log_info(f">>> Starting to process files for {year}-{current_month:02d}")

                current_day = timestamp.date()
                if prev_date is not None and current_day != prev_date:
                    processor.process_daystocks(prev_date)
                    day_committed = processor.commit_daystocks()
                    daystocks_added_total += day_committed

                # Process the file
                companies_added, stocks_processed = processor.process_boursorama_file(file_path)

                companies_added_total += companies_added
                stocks_in_current_commit_batch += stocks_processed
                stocks_added_total += stocks_processed # Accumulate total count

                if companies_added > 0 or stocks_processed > 0:
                    files_processed += 1
                    files_processed_in_month += 1

                    # Commit periodically based on stock count
                    if stocks_in_current_commit_batch >= commit_threshold:
                        stocks_committed = processor.commit_stocks()
                        log_info(f">>> Progress: Committed {stocks_committed} stock records, {files_processed} files processed")
                        stocks_in_current_commit_batch = 0 # Reset batch counter

                processed_files_cache.add(file_path)
                processed_files.add(file_path) # Update global set
                prev_date = current_day

            if last_month_logged is not None:
                log_info(f">>> Completed processing {files_processed_in_month} files for {year}-{last_month_logged:02d}")

        log_info(f"Completed processing year {year} with {files_processed} files so far")

    # Process final day's stocks
    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added_total += day_committed

    # Final cleanup and commit
    log_info("Cleaning stocks data and performing final commit...")
    processor.clean_stocks()
    final_companies, final_stocks, final_days = processor.commit_all()

    # Adjust total counts based on final commit
    companies_added_total += final_companies # Add companies committed at the end
    # stocks_added_total is already accumulated
    daystocks_added_total += final_days # Add daystocks committed at the end

    log_info(f"Boursorama processing complete: {files_processed} files processed, "
             f"{companies_added_total} companies added, {stocks_added_total} stocks added, "
             f"{daystocks_added_total} daystocks added")

    return files_processed, companies_added_total, stocks_added_total, daystocks_added_total


@timer_decorator
def process_euronext_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Euronext"""
    log_info(f"Processing Euronext files from {start_date.date()} to {end_date.date()}")

    processor = Processor(db_model) # Use a separate processor instance if needed, or pass the main one
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    directory_path = os.path.join(HOME, "euronext")

    files_to_process = []
    if os.path.exists(directory_path):
        try:
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    if not entry.is_file(): continue
                    file_path = entry.path
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', entry.name)
                    if date_match:
                        try:
                            file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                            if start_date <= file_date <= end_date:
                                files_to_process.append((file_date, file_path))
                        except ValueError:
                            log_error(f"Invalid date format in Euronext filename: {entry.name}")
        except FileNotFoundError:
             log_error(f"Euronext directory not found: {directory_path}")


    files_to_process.sort() # Sort by date
    files_to_process_paths = [f[1] for f in files_to_process] # Get only paths

    log_info(f"Total Euronext files to process: {len(files_to_process_paths)}")

    files_processed = 0
    companies_added_total = 0
    stocks_added_total = 0
    daystocks_added_total = 0
    prev_date = None
    stocks_in_current_commit_batch = 0 # Track for commit threshold

    for file_path in files_to_process_paths:
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        file_date = None
        if date_match:
             try:
                 file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
             except ValueError:
                 log_error(f"Skipping Euronext file due to invalid date: {filename}")
                 continue


        if file_date:
            current_day = file_date.date()
            if prev_date is not None and current_day != prev_date:
                processor.process_daystocks(prev_date)
                day_committed = processor.commit_daystocks()
                daystocks_added_total += day_committed

            df = load_euronext_file(file_path)
            if df is not None and not df.empty:
                companies_added = processor.process_dataframe(df)
                stocks_processed = len(df) # Number of rows in the loaded dataframe

                companies_added_total += companies_added
                stocks_added_total += stocks_processed
                stocks_in_current_commit_batch += stocks_processed
                files_processed += 1

                # Commit periodically
                if stocks_in_current_commit_batch >= 50000: # Use same threshold
                     stocks_committed = processor.commit_stocks()
                     log_info(f">>> Progress (Euronext): Committed {stocks_committed} stock records")
                     stocks_in_current_commit_batch = 0


            prev_date = current_day

    # Process final day's stocks
    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added_total += day_committed

    # Final cleanup and commit for Euronext data
    processor.clean_stocks() # Clean remaining stocks in batch
    final_companies, final_stocks, final_days = processor.commit_all()

    companies_added_total += final_companies
    # stocks_added_total is already accumulated
    daystocks_added_total += final_days

    log_info(f"Euronext processing complete: {files_processed} files processed, "
             f"{companies_added_total} companies added, {stocks_added_total} stocks added, "
             f"{daystocks_added_total} daystocks added")

    return files_processed, companies_added_total, stocks_added_total, daystocks_added_total


@timer_decorator
def clean_database(db_model):
    """Nettoie et optimise la base de données"""
    log_info("Cleaning database...")
    try:
        # 1. Supprimer les valeurs invalides
        db_model.execute("DELETE FROM stocks WHERE value <= 0 OR value IS NULL OR value > 100000")

        # 2. Corriger les volumes négatifs (Should not happen with current logic, but good safeguard)
        db_model.execute("UPDATE stocks SET volume = 0 WHERE volume < 0")

        # 3. Supprimer les actions orphelines
        db_model.execute("DELETE FROM stocks WHERE cid NOT IN (SELECT id FROM companies)")

        db_model.commit()

        # 4. Optimiser la base de données (créer des index)
        log_info("Creating indexes (if not exist)...")
        db_model.execute("CREATE INDEX IF NOT EXISTS idx_stocks_date_cid ON stocks(date, cid)")
        db_model.execute("CREATE INDEX IF NOT EXISTS idx_companies_symbol ON companies(symbol)")
        db_model.execute("CREATE INDEX IF NOT EXISTS idx_companies_isin ON companies(isin)")
        # Consider index on daystocks as well
        db_model.execute("CREATE INDEX IF NOT EXISTS idx_daystocks_date_cid ON daystocks(date, cid)")

        db_model.commit()
        log_info("Database cleanup and indexing complete")
    except Exception as e:
        log_error(f"Database cleaning failed: {str(e)}")
        if hasattr(db_model, 'connection') and db_model.connection:
            try:
                db_model.connection.rollback()
            except Exception as rb_err:
                log_error(f"Rollback failed during cleanup: {rb_err}")


#=================================================
# SECTION 5: FONCTION PRINCIPALE
#=================================================

@timer_decorator
def main():
    """Fonction ETL principale"""

    log_info("="*50)
    log_info("STARTING ETL PROCESS")
    log_info("="*50)

    db = None # Initialize db to None
    try:
        # Se connecter à la base de données
        log_info("Connecting to database...")
        db = tsdb.TimescaleStockMarketModel(
            database='bourse',
            user='ricou',
            host='db',
            password='monmdp'
        )

        # Réinitialiser la base de données
        log_info("Reinitializing database...")
        db._purge_database()
        db._setup_database()

        # Tester la connexion
        test_result = db.df_query("SELECT 1 AS test")
        if test_result is None or test_result.empty:
            log_error("Database connection test failed")
            return 1
        log_info("Database connection established")

        # Définir les plages de dates
        start_date = datetime(2020, 5, 1)
        end_date = datetime(2020, 7, 31)
        log_info(f"Using date range: {start_date.date()} to {end_date.date()}")

        # Traiter les fichiers Boursorama
        process_boursorama_files(db, start_date, end_date)

        # Traiter les fichiers Euronext
        process_euronext_files(db, start_date, end_date)

        # Nettoyer la base de données
        clean_database(db)

        # Résumé final
        log_info("\nFetching final counts...")
        companies_df = db.df_query("SELECT COUNT(*) as count FROM companies")
        stocks_df = db.df_query("SELECT COUNT(*) as count FROM stocks")
        daystocks_df = db.df_query("SELECT COUNT(*) as count FROM daystocks")
        companies_with_stocks = db.df_query("SELECT COUNT(DISTINCT cid) as count FROM stocks") # Alias count

        companies_count = companies_df['count'].iloc[0] if not companies_df.empty else 0
        stocks_count = stocks_df['count'].iloc[0] if not stocks_df.empty else 0
        daystocks_count = daystocks_df['count'].iloc[0] if not daystocks_df.empty else 0
        with_stocks_count = companies_with_stocks['count'].iloc[0] if not companies_with_stocks.empty else 0

        log_info("\n" + "="*50)
        log_info("ETL COMPLETION SUMMARY")
        log_info("="*50)
        log_info(f"Companies in database: {companies_count}")
        log_info(f"Companies with stock data: {with_stocks_count}")
        log_info(f"Stock records in database: {stocks_count}")
        log_info(f"Daily stock records in database: {daystocks_count}")

        # Imprimer les statistiques de chronométrage
        log_info("\nTiming Summary:")
        for func, duration in time_stats.items():
            log_info(f"{func}: {duration:.2f} seconds")

        log_info("\nETL process completed successfully!")
        log_info("="*50)

    except Exception as e:
        log_error(f"ETL process failed: {str(e)}")
        traceback.print_exc()
        return 1
    finally:
        # Ensure database connection is closed
        if db and hasattr(db, 'close'):
            log_info("Closing database connection.")
            db.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())
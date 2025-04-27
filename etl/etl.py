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
import bz2  

# Configuration de base
HOME = "/home/bourse/data/"  # Répertoires boursorama et euronext attendus
logger = mylogging.getLogger(__name__, filename="/tmp/bourse.log")

# Pour print direct sous docker
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Statistiques et suivi du traitement
time_stats = {}

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

def normalize_symbol_and_market(symbol, mid=6, cache=None): 
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

    if cache is not None:
        cache_key = (symbol, mid)
        if cache_key in cache:
            return cache[cache_key]

    result = None

    for prefix, (_, market_id) in MARKET_PREFIXES.items():
        if symbol.startswith(prefix):
            normalized_symbol = symbol[len(prefix):]
            result = (normalized_symbol, market_id)
            break 

    if result is None:
        result = (symbol, mid)

    if cache is not None:
        cache[cache_key] = result

    return result

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
        self.symbol_normalization_cache = {} # Cache des symbols car souvent redondants (la fonction normalize est appelée au moins 4 millions de fois selon snakeviz)
        self.isin_to_id_map = {}    # Map des ISINs vers IDs d'entreprises

        # Charger le cache de marché
        self._load_market_cache()

    def _load_market_cache(self):
        """Charge les IDs de marché depuis la base de données"""
        markets_df = self.db.df_query("SELECT id, alias FROM markets")
        self.market_cache = dict(zip(markets_df['alias'], markets_df['id'])) if not markets_df.empty else {}
        log_info(f"Market cache loaded with {len(self.market_cache)} markets") # Toujours 10

    def get_market_id(self, market_alias):
        """Obtient l'ID de marché depuis l'alias, avec repli vers Paris"""
        return self.market_cache.get(market_alias, 6)  # Par défaut Paris (ID 6)

    def process_boursorama_file(self, file_path):
        """Traite un fichier Boursorama individuel (normal ou .bz2)"""

        try:
            filename = os.path.basename(file_path)

            if filename.endswith('.bz2'):
                base_filename = filename[:-4]
                parts = base_filename.split(' ', 1)
            else:
                parts = filename.split(' ', 1)

            alias = parts[0]
            date_str = parts[1]

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

            df = None
            try:
                if filename.endswith('.bz2'):
                    with bz2.open(file_path, 'rb') as f:
                        df = pd.read_pickle(f)
                else:
                    df = pd.read_pickle(file_path)
            except Exception as read_error:
                log_error(f"Error reading file {file_path}: {str(read_error)}")
                return 0, 0

            # Gérer le cas où le symbole est à la fois index et colonne
            if df.index.name == 'symbol' and 'symbol' in df.columns:
                df = df.drop(columns=['symbol'])

            if df.index.name == 'symbol':
                df = df.reset_index()

            mid = self.get_market_id(alias)
            df["mid"] = mid
            df["date"] = timestamp

            df['name'] = df['name'].astype(str).str.removeprefix('SRD')
                            
            # Last est pas le même entre bz2 et les pickle de 2024
            if pd.api.types.is_numeric_dtype(df['last']):
                pass
            else:
                if df['last'].dtype == object:  
                    try:
                        df['last'] = df['last'].str.replace(r'\([a-zA-Z]\)|\s+', '', regex=True)
                    except AttributeError:
                        # Pas sur de l'utilité pas au cas ou
                        df['last'] = df['last'].astype(str).replace(r'\([a-zA-Z]\)|\s+', '', regex=True)
            
            df['last'] = pd.to_numeric(df['last'], errors='coerce')
            
            # On enlève les valeurs inutiles pour alléger et augmenté la rapidité
            valid_mask = (df['last'] > 0) & (df['volume'] > 0)
            df = df[valid_mask]

            companies_added = self.process_dataframe(df)

            return companies_added, len(df)

        except Exception as e:
            log_error(f"Error processing Boursorama file {file_path}: {str(e)}")
            return 0, 0

    def process_dataframe(self, df):
        """
        Traite le DataFrame d'un fichier en extrayant ses données et en les mettant
        dans les tables d'entreprise et d'actions.
        """
        companies_df = self.__process_companies(df)
        companies_added = 0
        if companies_df is not None and not companies_df.empty:
             companies_added = len(companies_df)
             self.companies_batch.append(companies_df)

        stocks_df = self.__process_stocks(df)

        if stocks_df is not None and not stocks_df.empty:
            self.stocks_batch.append(stocks_df)
            self.day_batch.append(stocks_df)

        return companies_added

    def __process_companies(self, df):
        """Traite le DataFrame et remplit la table des entreprises"""
        companies = df[['symbol', 'name', 'mid']].drop_duplicates('symbol')

        has_isin = 'isin' in df.columns
        if has_isin:
            companies['isin'] = df['isin'].copy()

        if companies.empty: # Peut être null si on a process des entreprises déjà traitées
            return pd.DataFrame()

        symbols = companies['symbol'].values
        mids = companies['mid'].values

        normalized_symbols = []
        normalized_mids = []

        for i, (symbol, mid) in enumerate(zip(symbols, mids)): # Important pour boursorama (pas trop pour euronext mais on le fais quand même pour simplicité de code)
            norm_symbol, norm_mid = normalize_symbol_and_market(symbol, mid, self.symbol_normalization_cache)
            normalized_symbols.append(norm_symbol)
            normalized_mids.append(norm_mid)

        companies['symbol'] = normalized_symbols
        companies['mid'] = normalized_mids

        companies = companies.drop_duplicates('symbol')
        companies.set_index('symbol', inplace=True)

        # Les 2 prochaines parties sont un peu "lourdes" mais elles sont nécessaires pour la gestion des ISINs
        # 1. On instancie le DataFrame companies_save si il n'existe pas pour la prochaine partie, le but est        # d'avoir un DataFrame avec les entreprises déjà présentes dans la base de données
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

        # 2. Mettre à jour les entreprises existantes avec les nouvelles données comme les noms et symboles peuvent changer entre euronext et boursorama
        if has_isin and self.isin_to_id_map:
            for idx, row in companies.iterrows():
                 if 'isin' in row and pd.notna(row['isin']) and row['isin'] in self.isin_to_id_map:
                    existing_id = self.isin_to_id_map[row['isin']]
                    if idx not in self.companies_save.index:
                        company_data = {'id': existing_id, 'name': row['name'], 'mid': row['mid'], 'isin': row['isin']}
                        self.companies_save.loc[idx] = company_data
                        update_query = "UPDATE companies SET name = %s, symbol = %s WHERE id = %s"
                        self.db.execute(update_query, [row['name'], idx, existing_id])

        # 3. On identifie les nouvelles entreprises à ajouter
        existing_symbols = set(self.companies_save.index)
        current_symbols = set(companies.index)
        new_symbols = current_symbols - existing_symbols

        if not new_symbols: # Si pas de nouvelles entreprises, on ne fait rien
            return pd.DataFrame()

        new_companies = companies.loc[list(new_symbols)].copy()

        if not new_companies.empty:
            next_id = 1
            if not self.companies_save.empty and 'id' in self.companies_save.columns:
                max_id = self.companies_save['id'].max() 
                if pd.notna(max_id):
                    next_id = int(max_id) + 1
            
            new_companies['id'] = np.arange(next_id, next_id + len(new_companies), dtype=np.int16)
            
            if has_isin:
                valid_isins = new_companies[new_companies['isin'].notna()]
                if not valid_isins.empty:
                    self.isin_to_id_map.update(dict(zip(valid_isins['isin'], valid_isins['id'])))
            
            self.companies_save = pd.concat([self.companies_save, new_companies])

        return new_companies

    def __process_stocks(self, df):
        """Traite le DataFrame et remplit la table des actions"""
        stocks_subset = df[['symbol', 'last', 'date', 'volume']].copy() #On créer une copie pour éviter des modifications sur le DataFrame d'origine
        stocks_subset.rename(columns={'last': 'value'}, inplace=True)        
        stocks_subset = stocks_subset[stocks_subset['volume'] > 0]
        
        default_mid = 6
        mids = df['mid'] if 'mid' in df.columns else [default_mid] * len(stocks_subset)
        
        normalized_symbols = []
        for symbol, mid in zip(stocks_subset['symbol'], mids): # On enlève les prefix
            norm_symbol, _ = normalize_symbol_and_market(symbol, mid, self.symbol_normalization_cache)
            normalized_symbols.append(norm_symbol)
        
        stocks_subset['normalized_symbol'] = normalized_symbols
        
        # 4. Map normalized symbols to company IDs
        symbol_to_cid_map = self.companies_save['id']
        stocks_subset['cid'] = stocks_subset['normalized_symbol'].map(symbol_to_cid_map)
                
        # 6. Create final DataFrame with correct types
        stocks_df = pd.DataFrame({
            'cid': stocks_subset['cid'].astype(np.int16),
            'date': stocks_subset['date'],
            'value': stocks_subset['value'],
            'volume': stocks_subset['volume'].astype(np.int32)  #On met le volume en int32 pour opti
        })
        
        stocks_df = stocks_df[stocks_df['value'] > 0].set_index('cid')
        
        return stocks_df

    def process_daystocks(self, date):
        """
        À partir du lot d'actions traité, crée des actions quotidiennes
        """
        if not self.day_batch:
            return 0
        
        daystocks = pd.concat(self.day_batch)

        # L'agrégation transforme les données granulaires (potentiellement des centaines par jour) en une synthèse OHLCV journalière par titre
        # Ici c'est super important pour la mémoire car ça réduit le volume de données à stocker et traiter
        aggregated_daystocks = daystocks.groupby(level='cid').agg( 
            open=pd.NamedAgg(column='value', aggfunc='first'),
            close=pd.NamedAgg(column='value', aggfunc='last'),
            high=pd.NamedAgg(column='value', aggfunc='max'),
            low=pd.NamedAgg(column='value', aggfunc='min'),
            volume=pd.NamedAgg(column='volume', aggfunc='sum')
        )

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

        stocks = pd.concat(self.stocks_batch) 

        stocks.reset_index(inplace=True)

        stocks.sort_values(['cid', 'date'], inplace=True, kind='mergesort')
        stocks['day'] = stocks['date'].dt.floor('D')

        # Calculer les différences entre valeurs consécutives
        # On utilise groupby() et shift() poru des meilleurs performances
        stocks['value_prev'] = stocks.groupby('cid')['value'].shift(1)
        stocks['value_change'] = (stocks['value'] - stocks['value_prev']).abs()
        # On empêche la division par zéro
        stocks['pct_change'] = (stocks['value_change'] / stocks['value_prev'].abs().replace(0, np.nan))

        # Garder les lignes avec changement significatif (plus de 0.1%)
        min_change_pct = 0.001 
        
        has_change = stocks['pct_change'].fillna(0) > min_change_pct

        # Ajouter les première et dernière valeurs de chaque jour
        
        first_of_day = ~stocks.duplicated(['cid', 'day'])
        last_of_day = ~stocks.duplicated(['cid', 'day'], keep='last')

        # Ajouter les première et dernière valeurs pour chaque ID d'entreprise
        first_of_cid = ~stocks.duplicated(['cid'])
        last_by_cid = ~stocks.duplicated(['cid'], keep='last')

        # Combiner les critères
        keep_mask = has_change | first_of_day | last_of_day | first_of_cid | last_by_cid

        # Appliquer le filtre et nettoyer
        stocks = stocks.loc[keep_mask, ['cid', 'date', 'value', 'volume']].copy() 
        stocks.set_index('cid', inplace=True)

        self.stocks_batch = [stocks] if not stocks.empty else []


    def commit_companies(self):
        """Enregistrer toutes les entreprises dans la base de données"""
        total_committed = 0
        
        if not self.companies_batch:
            return 0

        all_new_companies = pd.concat(self.companies_batch)

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


        self.companies_batch = []

        return total_committed


    def commit_stocks(self):
        """Enregistrer toutes les actions dans la base de données"""

        if not self.stocks_batch:
            return 0

        total_committed = 0
        all_stocks = pd.concat(self.stocks_batch)

        self.db.df_write(
            all_stocks,
            'stocks',
            commit=True,
            index=True,
            index_label='cid'
        )
        total_committed = len(all_stocks)

        self.stocks_batch = []
        return total_committed


    def commit_daystocks(self):
        """Enregistrer toutes les actions quotidiennes dans la base de données"""
        if not self.daystocks_batch:
            return 0

        total_committed = 0
        all_daystocks = pd.concat(self.daystocks_batch)

         # Écrire dans la base de données
        self.db.df_write(all_daystocks, 'daystocks', commit=True, index=True, index_label='cid')
        total_committed = len(all_daystocks)

        self.daystocks_batch = []

        return total_committed


    def commit_all(self):
        """Enregistrer toutes les données en attente dans la base de données"""
        companies = self.commit_companies()
        stocks = self.commit_stocks()
        daystocks = self.commit_daystocks()
        return companies, stocks, daystocks

#=================================================
# SECTION 4: FONCTIONS DE TRAITEMENT DE FICHIERS
#=================================================

def load_euronext_file(file_path):
    """Charge et analyse un fichier Euronext"""
    try:
        df = None
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, delimiter='\t', skiprows=None,
                            on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)

        # Standardiser les noms de colonnes
        column_mapping = {
            'Symbol': 'symbol', 
            'Name': 'name', 
            'Last': 'last', 
            'last Price': 'last',
            'Volume': 'volume', 
            'ISIN': 'isin',
            'Market': 'market'
        }
        rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        df.rename(columns=rename_dict, inplace=True)

        df['name'] = df['name'].str.removeprefix('SRD')

        # Converti 'last' et 'volume' de manière efficace
        df['last'] = pd.to_numeric(df['last'].fillna('').astype(str).str.replace(r'[^\d.,]+', '', regex=True).str.replace(',', '.', regex=False), errors='coerce')

        df['volume'] = pd.to_numeric(df['volume'].astype(str).str.replace(r'[^\d]+', '', regex=True), errors='coerce').fillna(0)

        # Comme avant on enlève les valeurs inutiles
        valid_mask = (df['volume'] > 0) & (df['last'] > 0)
        df = df[valid_mask]

        # Extraire la date du nom de fichier
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            file_date = date_match.group(1)
            df['date'] = pd.to_datetime(file_date)
            
            df['mid'] = 6
            
            if 'market' in df.columns:
                market_map = {
                    'paris': 6,
                    'amsterdam': 5,
                    'london': 2,
                    'milan': 3,
                    'madrid': 4,
                    'brussels': 8,
                    'xetra': 7
                }
                
                def extract_market_id(market_str):                    
                    market_str = market_str.lower()
                    for market_name, market_id in market_map.items():
                        if market_name in market_str:
                            return market_id
                    return 6  
                
                df['mid'] = df['market'].apply(extract_market_id)
            
            return df

    except Exception as e:
        log_error(f"Error loading Euronext file {file_path}: {str(e)}")
        return None

def extract_boursorama_day(date_str):
    """
    Extrait la partie date (YYYY-MM-DD) d'une chaîne de date Boursorama.
    """
    try:
        return re.match(r'(\d{4}-\d{2}-\d{2})', date_str).group(1)
    except Exception:
        return None

@timer_decorator
def process_boursorama_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Boursorama
    
    Additional feature : 
       - Possibilité de choisir l'interval de date afin d'obtenir les stocks dans le temps qui nous intéresse au lieu de tout traiter.
    
    """
    log_info(f"Processing Boursorama files from {start_date.date()} to {end_date.date()}")

    processor = Processor(db_model) # load processor 
    
    start_date = pd.to_datetime(start_date) # Gestion de date personalisé
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
    stocks_added_total = 0 
    daystocks_added_total = 0
    prev_date = None
    date_cache = {}

    commit_threshold = 100000 # Changé en fonction de RAM, plus il est élévé moins de commit mais plus lourd en mémoire

    for year in years_to_process:
        year_dir = os.path.join(boursorama_path, str(year))
        log_info(f"Scanning files in year directory {year}")
        year_files = []

        # On utilise scandir plutôt que os.listdir pour une meilleure performance
        with os.scandir(year_dir) as entries:
            for entry in entries:
                file_path = entry.path
                filename = entry.name
                base_name = filename[:-4] if filename.endswith('.bz2') else filename
                parts = base_name.split(' ', 1)

                date_str = parts[1]
                day_part = extract_boursorama_day(date_str)

                month_str = day_part.split('-')[1]
                month = int(month_str)
                
                if day_part in date_cache:
                    timestamp = date_cache[day_part]
                else:
                    timestamp = pd.to_datetime(day_part)
                    date_cache[day_part] = timestamp

                    # Final date range check
                if start_date <= timestamp <= end_date:
                    year_files.append((timestamp, file_path, parts[0]))


        year_files.sort() 

        if year_files:
            months_count = {}
            for timestamp, _, _ in year_files:
                month = timestamp.month
                months_count[month] = months_count.get(month, 0) + 1
            log_info(f"Found {len(year_files)} files to process in year {year}")
            log_info(f"Files distribution by month in {year}: {months_count}")

            last_month_logged = None
            files_processed_in_month = 0
            stocks_in_current_commit_batch = 0 

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

                # On processe le fichier Boursorama
                companies_added, stocks_processed = processor.process_boursorama_file(file_path)

                companies_added_total += companies_added
                stocks_in_current_commit_batch += stocks_processed
                stocks_added_total += stocks_processed 

                if companies_added > 0 or stocks_processed > 0:
                    files_processed += 1
                    files_processed_in_month += 1

                    # Commit périodiquement pour éviter de surcharger la mémoire
                    if stocks_in_current_commit_batch >= commit_threshold:
                        stocks_committed = processor.commit_stocks()
                        log_info(f">>> Progress: Committed {stocks_committed} stock records, {files_processed} files processed")
                        stocks_in_current_commit_batch = 0 

                prev_date = current_day

            if last_month_logged is not None:
                log_info(f">>> Completed processing {files_processed_in_month} files for {year}-{last_month_logged:02d}")

        log_info(f"Completed processing year {year} with {files_processed} files so far")

    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added_total += day_committed

    log_info("Cleaning stocks data and performing final commit...")
    processor.clean_stocks()
    final_companies, final_stocks, final_days = processor.commit_all()

    companies_added_total += final_companies 
    daystocks_added_total += final_days 

    log_info(f"Boursorama processing complete: {files_processed} files processed, "
             f"{companies_added_total} companies added, {stocks_added_total} stocks added, "
             f"{daystocks_added_total} daystocks added")

    return files_processed, companies_added_total, stocks_added_total, daystocks_added_total


@timer_decorator
def process_euronext_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Euronext"""
    log_info(f"Processing Euronext files from {start_date.date()} to {end_date.date()}")

    processor = Processor(db_model)
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
                        file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                        if start_date <= file_date <= end_date:
                            files_to_process.append((file_date, file_path))
        except FileNotFoundError:
             log_error(f"Euronext directory not found, check if you have respected data architecture: {directory_path}")
             
    if not files_to_process:
        log_info("No Euronext files found in the specified date range.")
        return 0, 0, 0, 0

    files_to_process.sort() 
    files_to_process_paths = [f[1] for f in files_to_process]

    log_info(f"Total Euronext files to process: {len(files_to_process_paths)}")

    files_processed = 0
    companies_added_total = 0
    stocks_added_total = 0
    daystocks_added_total = 0
    prev_date = None
    stocks_in_current_commit_batch = 0 

    for file_path in files_to_process_paths:
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')

        if file_date:
            current_day = file_date.date()
            if prev_date is not None and current_day != prev_date:
                processor.process_daystocks(prev_date)
                day_committed = processor.commit_daystocks()
                daystocks_added_total += day_committed

            df = load_euronext_file(file_path)
            if df is not None and not df.empty:
                companies_added = processor.process_dataframe(df)
                stocks_processed = len(df) 

                companies_added_total += companies_added
                stocks_added_total += stocks_processed
                stocks_in_current_commit_batch += stocks_processed
                files_processed += 1

                # idem que pour boursorama
                if stocks_in_current_commit_batch >= 50000: 
                     stocks_committed = processor.commit_stocks()
                     log_info(f">>> Progress (Euronext): Committed {stocks_committed} stock records")
                     stocks_in_current_commit_batch = 0


            prev_date = current_day

    # Process le dernier jours
    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added_total += day_committed

    processor.clean_stocks()
    final_companies, final_stocks, final_days = processor.commit_all()

    companies_added_total += final_companies
    daystocks_added_total += final_days

    log_info(f"Euronext processing complete: {files_processed} files processed, "
             f"{companies_added_total} companies added, {stocks_added_total} stocks added, "
             f"{daystocks_added_total} daystocks added")

    return files_processed, companies_added_total, stocks_added_total, daystocks_added_total

# Pas nécessaire mais juste au cas ou !!!
@timer_decorator
def clean_database(db_model):
    """Nettoie et optimise la base de données"""
    log_info("Cleaning database...")
    # 1. Supprimer les valeurs invalides
    db_model.execute("DELETE FROM stocks WHERE value <= 0 OR value IS NULL OR value > 100000")

    # 2. Corriger les volumes négatifs
    db_model.execute("UPDATE stocks SET volume = 0 WHERE volume < 0")

    # 3. Supprimer les actions orphelines
    db_model.execute("DELETE FROM stocks WHERE cid NOT IN (SELECT id FROM companies)")

    db_model.commit()

    log_info("Database cleanup and indexing complete")

#=================================================
# SECTION 5: FONCTION PRINCIPALE
#=================================================

@timer_decorator
def main():
    """Fonction ETL principale"""

    log_info("="*50)
    log_info("STARTING ETL PROCESS")
    log_info("="*50)

    db = None 
    try:
        # Se connecter à la base de données
        log_info("Connecting to database...")
        db = tsdb.TimescaleStockMarketModel(
            database='bourse',
            user='ricou',
            host='db',
            password='monmdp'
        )


        log_info("Database connection established")

        # Définir les plages de dates, ici on traite tout de 2019 à 2024 par défaut mais le user peut sélectionner une plage de date (ADDITIONNAL FEATURE)
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
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
        companies_with_stocks = db.df_query("SELECT COUNT(DISTINCT cid) as count FROM stocks") 

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
        if db and hasattr(db, 'close'):
            log_info("Closing database connection.")
            db.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())
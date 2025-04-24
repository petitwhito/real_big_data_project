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
from io import BytesIO  # Ajout pour décompression en mémoire

# Configuration de base
HOME = "/home/bourse/data/"  # Répertoires boursorama et euronext attendus
logger = mylogging.getLogger(__name__, filename="/tmp/bourse.log")

# Forcer le vidage immédiat de la sortie pour les environnements Docker
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Statistiques et suivi du traitement
time_stats = {}
processed_files = set()

# Structure des statistiques de normalisation
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

def clean_company_name(name):
    """Nettoie les noms d'entreprises en supprimant le préfixe SRD"""
    if name and isinstance(name, str) and name.startswith('SRD'):
        return name[3:]
    return name

def normalize_symbol_and_market(symbol, mid=6):
    """
    Normalise un symbole Boursorama en supprimant le préfixe
    et détermine l'ID du marché correct
    
    Args:
        symbol: Chaîne de symbole originale
        mid: ID du marché original (défaut: 6 pour Paris)
    
    Returns:
        tuple: (symbole_normalisé, id_marché)
    """
    if not symbol or not isinstance(symbol, str):
        return symbol, mid
    
    # Mettre à jour le compteur global
    symbol_normalization_stats['symbols_processed'] += 1
    
    # Vérifier d'abord les cas spéciaux
    if symbol in SPECIAL_CASES:
        normalized, market_id = SPECIAL_CASES[symbol]
        symbol_normalization_stats['special_cases'] += 1
        return normalized, market_id
    
    # Vérifier le préfixe et le supprimer si trouvé
    for prefix, (_, market_id) in MARKET_PREFIXES.items():
        if symbol.startswith(prefix):
            symbol_normalization_stats['prefix_removed'] += 1
            return symbol[len(prefix):], market_id
    
    # Si aucun préfixe trouvé, renvoyer l'original
    return symbol, mid

# Fonction utilitaire pour lire les fichiers boursorama (normaux ou .bz2)
def read_boursorama_file(file_path):
    """Lit un fichier Boursorama, qu'il soit compressé (.bz2) ou non"""
    try:
        if file_path.endswith('.bz2'):
            with bz2.open(file_path, 'rb') as f:
                compressed_content = f.read()
            return pd.read_pickle(BytesIO(compressed_content))
        else:
            return pd.read_pickle(file_path)
    except Exception as e:
        log_error(f"Error reading file {file_path}: {str(e)}")
        return None

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
        self.symbol_normalization_cache = {}  # Cache pour les symboles normalisés
        self.isin_to_id_map = {}    # Nouvelle: Map des ISINs vers IDs d'entreprises
        
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
            # Pour les fichiers .bz2, le nom réel est sans l'extension
            if filename.endswith('.bz2'):
                base_filename = filename[:-4]  # Retirer '.bz2'
                parts = base_filename.split(' ', 1)
            else:
                parts = filename.split(' ', 1)
                
            if len(parts) < 2:
                log_error(f"Invalid filename format: {filename}")
                return 0, 0
                
            alias = parts[0]  # Alias du marché (ex: compA)
            date_str = parts[1]  # Partie date
            
            # Traiter les dates avec underscores (_)
            try:
                if '_' in date_str:
                    # Extraire la partie date (avant l'espace)
                    date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
                    
                    # Gérer les dates au format YYYY-MM-DD HH_MM_SS.ffffff
                    # Remplacer les underscores par des deux-points pour l'heure
                    hour_part = date_str.replace(date_part, '').strip()
                    if hour_part.startswith(' '):
                        hour_part = hour_part[1:]  # Enlever l'espace initial
                    
                    # Remplacer les underscores par des deux-points dans la partie heure
                    hour_part_fixed = hour_part.replace('_', ':')
                    
                    # Reconstruire la date complète
                    formatted_date_str = f"{date_part} {hour_part_fixed}"
                    timestamp = pd.to_datetime(formatted_date_str)
                else:
                    timestamp = pd.to_datetime(date_str)
                    
            except Exception as e:
                log_error(f"Error parsing date in {filename}: {str(e)}")
                # Utiliser uniquement la partie date (YYYY-MM-DD) comme fallback
                day_part = extract_boursorama_day(date_str)
                if day_part:
                    timestamp = pd.to_datetime(day_part)
                else:
                    raise ValueError(f"Unable to parse date from filename: {filename}")
            
            # Charger le fichier (gérer .bz2 et fichiers normaux)
            if filename.endswith('.bz2'):
                with bz2.open(file_path, 'rb') as f:
                    compressed_content = f.read()
                df = pd.read_pickle(BytesIO(compressed_content))
            else:
                df = pd.read_pickle(file_path)
                
            if df is None or df.empty:
                return 0, 0
            
            # Gérer le cas où le symbole est à la fois index et colonne
            if df.index.name == 'symbol' and 'symbol' in df.columns:
                df = df.drop(columns=['symbol'])
                
            # Réinitialiser l'index s'il est nommé 'symbol'
            if df.index.name == 'symbol':
                df = df.reset_index()
                    
            # Si le symbole n'est toujours pas une colonne, nous devons l'ajouter
            if 'symbol' not in df.columns:
                log_error(f"No symbol column in {filename}")
                return 0, 0
                    
            # Obtenir l'ID de marché pour cet alias
            mid = self.get_market_id(alias)
            
            # Préparer le DataFrame
            df["mid"] = mid
            df["date"] = timestamp
            
            # AMÉLIORATION: Nettoyer les noms (supprimer préfixe SRD)
            if 'name' in df.columns:
                df['name'] = df['name'].apply(clean_company_name)
            
            # S'assurer que les colonnes requises existent
            if 'name' not in df.columns:
                df['name'] = df['symbol']  # Utiliser le symbole comme nom s'il manque
                
            # Traiter la colonne 'last'
            if 'last' in df.columns:
                # Nettoyer la colonne 'last' si c'est un type chaîne
                if df['last'].dtype == 'object':  # Valeurs de chaîne
                    df['last'] = df['last'].str.replace(r'\([a-zA-Z]\)|\s+', '', regex=True)
                    # Gérer notation européenne
                    df['last'] = df['last'].str.replace(',', '.', regex=False)
                
                # Convertir en numérique
                df['last'] = pd.to_numeric(df['last'], errors='coerce')
            else:
                # AMÉLIORATION: Essayer d'autres colonnes de prix si 'last' n'existe pas
                if 'value' in df.columns:
                    df['last'] = pd.to_numeric(df['value'], errors='coerce')
                elif 'close' in df.columns:
                    df['last'] = pd.to_numeric(df['close'], errors='coerce')
                else:
                    log_error(f"No price column (last/value/close) in {filename}")
                    return 0, 0
                
            # Ajouter le volume s'il manque
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                # AMÉLIORATION: Meilleur traitement des volumes
                if df['volume'].dtype == 'object':
                    df['volume'] = df['volume'].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            
            # AMÉLIORATION: Filtrer les volumes nuls et valeurs invalides
            valid_mask = (df['last'] > 0) & (df['volume'] > 0)
            df = df[valid_mask].dropna(subset=['last'])
        
            if df.empty:
                return 0, 0
            
            # Traiter le DataFrame
            companies_added = self.process_dataframe(df)
            
            # Marquer comme traité
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
        companies_added = len(companies_df)
        
        # Traiter les actions
        stocks_df = self.__process_stocks(df)
        
        # Ajouter aux lots
        if not companies_df.empty:
            self.companies_batch.append(companies_df)
        
        if not stocks_df.empty:
            self.stocks_batch.append(stocks_df)
            self.day_batch.append(stocks_df)
        
        return companies_added
    
    def __process_companies(self, df):
        """Traite le DataFrame et remplit la table des entreprises - Version ultra-optimisée"""
        # Vérification rapide des colonnes requises
        if not {'symbol', 'name'}.issubset(df.columns):
            return pd.DataFrame()
        
        # Optimisation 1: Utiliser une approche plus directe pour la normalisation des symboles
        # Créer une copie uniquement des colonnes nécessaires
        companies = df[['symbol', 'name', 'mid']].drop_duplicates('symbol').dropna(subset=['symbol'])
        
        # AMÉLIORATION: Extraire et traiter l'ISIN s'il est disponible
        has_isin = 'isin' in df.columns
        if has_isin:
            companies['isin'] = df['isin'].copy()
        
        if companies.empty:
            return pd.DataFrame()
        
        # Optimisation 2: Utiliser NumPy pour la normalisation en lot
        symbols = companies['symbol'].values
        mids = companies['mid'].values
        
        # Vecteur pour stocker les résultats
        normalized_symbols = []
        normalized_mids = []
        
        # Normaliser chaque symbole
        for i, (symbol, mid) in enumerate(zip(symbols, mids)):
            norm_symbol, norm_mid = normalize_symbol_and_market(symbol, mid)
            normalized_symbols.append(norm_symbol)
            normalized_mids.append(norm_mid)
        
        # Assigner les valeurs normalisées
        companies['symbol'] = normalized_symbols
        companies['mid'] = normalized_mids
        
        # Optimisation 3: Déduplication après normalisation
        companies = companies.drop_duplicates('symbol')
        companies.set_index('symbol', inplace=True)
        
        # Gestion optimisée du cache d'entreprises
        if self.companies_save is None:
            existing_companies = self.db.df_query("SELECT id, name, symbol, mid, isin FROM companies")
            
            if not existing_companies.empty:
                existing_companies.set_index('symbol', inplace=True)
                self.companies_save = existing_companies
                
                # AMÉLIORATION: Construire le mappage ISIN vers ID
                if 'isin' in existing_companies.columns:
                    isin_map = existing_companies[~existing_companies['isin'].isna()][['id', 'isin']]
                    self.isin_to_id_map = dict(zip(isin_map['isin'], isin_map['id']))
            else:
                self.companies_save = pd.DataFrame(columns=['id', 'name', 'mid', 'isin'])
                self.companies_save.index.name = 'symbol'
        
        # AMÉLIORATION: Vérifier si des entreprises avec ISIN correspondent à des entreprises existantes
        if has_isin and self.isin_to_id_map:
            for idx, row in companies.iterrows():
                if 'isin' in row and pd.notna(row['isin']) and row['isin'] in self.isin_to_id_map:
                    # Entreprise avec ISIN déjà connue mais nom/symbole différent
                    existing_id = self.isin_to_id_map[row['isin']]
                    # Mettre à jour le cache et la base
                    if idx not in self.companies_save.index:
                        # Nouvelle entrée - copier les données existantes mais avec nouveau symbole
                        company_data = {'id': existing_id, 'name': row['name'], 'mid': row['mid'], 'isin': row['isin']}
                        self.companies_save.loc[idx] = company_data
                        
                        # Mettre à jour dans la base
                        update_query = """
                        UPDATE companies 
                        SET name = %s, symbol = %s 
                        WHERE id = %s"""
                        self.db.execute(update_query, [row['name'], idx, existing_id])
        
        # Optimisation 4: Utiliser des ensembles pour opérations d'ensemble très rapides
        existing_symbols = set(self.companies_save.index)
        current_symbols = set(companies.index)
        new_symbols = current_symbols - existing_symbols
        
        # Filtrer pour obtenir seulement les nouvelles entreprises
        if not new_symbols:
            return pd.DataFrame()
        
        new_companies = companies.loc[list(new_symbols)].copy()
        
        if not new_companies.empty:
            # Optimisation 5: Génération d'ID efficace
            next_id = 1
            if not self.companies_save.empty and 'id' in self.companies_save.columns:
                next_id = min(32700, self.companies_save['id'].max() + 1)
            
            # Vérifier les limites SMALLINT
            if next_id + len(new_companies) > 32767:
                new_companies = new_companies.iloc[:(32767 - next_id)]
                
                if new_companies.empty:
                    return pd.DataFrame()
            
            # Création efficace des IDs avec numpy
            new_companies['id'] = np.arange(next_id, next_id + len(new_companies), dtype=np.int16)
            
            # AMÉLIORATION: Mettre à jour le mappage ISIN vers ID
            if has_isin:
                for idx, row in new_companies.iterrows():
                    if 'isin' in row and pd.notna(row['isin']):
                        self.isin_to_id_map[row['isin']] = row['id']
            
            # Mise à jour du cache
            self.companies_save = pd.concat([self.companies_save, new_companies])
        
        return new_companies
    
    def __process_stocks(self, df):
        """Traite le DataFrame et remplit la table des actions - Version ultra-optimisée"""
        # Vérifications préliminaires rapides
        if self.companies_save is None or self.companies_save.empty:
            return pd.DataFrame()
        
        if not {'symbol', 'date'}.issubset(df.columns):
            return pd.DataFrame()
        
        # Optimisation 1: Gérer last/value/close de manière vectorisée
        if 'last' not in df.columns:
            if 'value' in df.columns:
                df['last'] = df['value']
            elif 'close' in df.columns:
                df['last'] = df['close']
            else:
                return pd.DataFrame()
        
        # Optimisation 2: Extraction efficace des données clés
        # Extraire seulement les colonnes nécessaires
        stocks_subset = df[['symbol', 'last', 'date']].copy()
        
        # Optimisation 3: Ajouter le volume si manquant
        if 'volume' in df.columns:
            stocks_subset['volume'] = df['volume']
        else:
            stocks_subset['volume'] = 0
        
        # AMÉLIORATION: Filtrer les volumes nuls
        stocks_subset = stocks_subset[stocks_subset['volume'] > 0]
        
        if stocks_subset.empty:
            return pd.DataFrame()
        
        # Optimisation 4: Normalisation efficace des symboles
        stocks_subset['normalized_symbol'] = stocks_subset['symbol'].apply(
            lambda s: normalize_symbol_and_market(s)[0]
        )
        
        # Optimisation 5: Mappage efficace des symboles vers les IDs
        companies_with_ids = self.companies_save.reset_index()[['symbol', 'id']].rename(
            columns={'id': 'cid'})
        
        # Créer un dictionnaire de mappage pour une recherche plus rapide
        symbol_to_cid = dict(zip(companies_with_ids['symbol'], companies_with_ids['cid']))
        
        # Mapper les symboles normalisés aux IDs de société
        # Optimisation 6: Utiliser une correspondance de dictionnaire (très rapide)
        cids = []
        for sym in stocks_subset['normalized_symbol']:
            cids.append(symbol_to_cid.get(sym, None))
        
        stocks_subset['cid'] = cids
        
        # Filtrer les lignes sans correspondance
        stocks_subset = stocks_subset.dropna(subset=['cid'])
        
        if stocks_subset.empty:
            return pd.DataFrame()
        
        # Optimisation 7: Conversion des types pour économiser la mémoire
        stocks_df = pd.DataFrame({
            'cid': stocks_subset['cid'].astype(np.int16),
            'date': stocks_subset['date'],
            'value': pd.to_numeric(stocks_subset['last'], errors='coerce'),
            'volume': pd.to_numeric(stocks_subset['volume'], errors='coerce').fillna(0).astype(np.int32)
        })
        
        # Optimisation 8: Filtrage plus efficace des valeurs invalides
        stocks_df = stocks_df[stocks_df['value'] > 0].dropna(subset=['value'])
        
        # Définir cid comme index
        stocks_df = stocks_df.set_index('cid')
        
        return stocks_df
    
    def process_daystocks(self, date):
        """
        À partir du lot d'actions traité, crée des actions quotidiennes
        """
        if not self.day_batch:
            return
        
        # Concaténer toutes les actions de ce jour
        # Optimisation: Utiliser concat avec copy=False pour économiser la mémoire
        daystocks = pd.concat(self.day_batch, copy=False)
        
        if daystocks.empty:
            self.day_batch = []  # Réinitialiser le lot journalier
            return
        
        # Optimisation: Utiliser agg avec NamedAgg pour une agrégation efficace
        daystocks = daystocks.groupby(['cid']).agg(
            open=pd.NamedAgg(column='value', aggfunc='first'),
            close=pd.NamedAgg(column='value', aggfunc='last'),
            high=pd.NamedAgg(column='value', aggfunc='max'),
            low=pd.NamedAgg(column='value', aggfunc='min'),
            volume=pd.NamedAgg(column='volume', aggfunc='sum')
        )
        
        # Définir la date
        daystocks['date'] = date
        
        # AMÉLIORATION: Ajouter des statistiques supplémentaires
        daystocks['mean'] = (daystocks['open'] + daystocks['close'] + daystocks['high'] + daystocks['low']) / 4
        
        # Ajouter au lot daystocks
        self.daystocks_batch.append(daystocks)
        
        # Nettoyer le lot journalier
        self.day_batch = []
    
    def clean_stocks(self):
        """Nettoyage optimisé des données de stocks"""
        if not self.stocks_batch:
            return
        
        # Combiner tous les lots avec copy=False pour économiser la mémoire
        stocks = pd.concat(self.stocks_batch, copy=False)
        
        if stocks.empty:
            self.stocks_batch = []
            return
        
        # Reset index et tri
        stocks.reset_index(inplace=True)
        
        # Optimisation 1: Utiliser sort_values avec inplace et algorithme mergesort (plus rapide que quicksort ici)
        stocks.sort_values(['cid', 'date'], inplace=True, kind='mergesort')
        
        # Optimisation 2: Conversion plus efficace du jour
        stocks['day'] = stocks['date'].dt.floor('D')
        
        # AMÉLIORATION: Optimisation du stockage pour réduire la taille
        # Calculer les différences entre valeurs consécutives
        stocks['value_prev'] = stocks.groupby('cid')['value'].shift(1)
        stocks['value_change'] = (stocks['value'] - stocks['value_prev']).abs()
        stocks['pct_change'] = stocks['value_change'] / stocks['value_prev'].abs()
        
        # Garder les lignes avec changement significatif (plus de 0.1%)
        min_change_pct = 0.001  # 0.1%
        has_change = stocks['pct_change'] > min_change_pct
        
        # Ajouter les première et dernière valeurs de chaque jour
        first_of_day = ~stocks.duplicated(['cid', 'day'])
        last_of_day = ~stocks.duplicated(['cid', 'day'], keep='last')
        
        # Ajouter les première et dernière valeurs pour chaque ID d'entreprise
        first_of_cid = ~stocks.duplicated(['cid'])
        last_by_cid = ~stocks.duplicated(['cid'], keep='last')
        
        # Combiner les critères
        keep_mask = has_change | first_of_day | last_of_day | first_of_cid | last_by_cid
        
        # Appliquer le filtre et nettoyer
        stocks = stocks[keep_mask]
        stocks.drop(columns=['value_prev', 'value_change', 'pct_change', 'day'], inplace=True)
        stocks.set_index('cid', inplace=True)
        
        # Remplacer le lot
        self.stocks_batch = [stocks]
    
    def commit_companies(self):
        """Enregistrer toutes les entreprises dans la base de données"""
        if not self.companies_batch:
            return 0
            
        total_committed = 0
        
        # Obtenir les IDs d'entreprise existants pour éviter les doublons
        existing_ids_df = self.db.df_query("SELECT id FROM companies")
        existing_ids = set(existing_ids_df['id'].tolist()) if not existing_ids_df.empty else set()
        
        for batch in self.companies_batch:
            if batch.empty:
                continue
            
            # Préparer pour l'insertion dans la base de données
            companies_df = batch.reset_index()
            
            # Filtrer les entreprises qui n'existent pas déjà
            # Optimisation: Utiliser un masque vectorisé au lieu d'une boucle
            mask = ~companies_df['id'].isin(existing_ids)
            new_companies = companies_df[mask]
            
            if not new_companies.empty:
                try:
                    # Utiliser la méthode du modèle DB pour l'insertion efficace
                    self.db.df_write(
                        new_companies,
                        'companies',
                        commit=True,
                        if_exists="append",
                        index=False
                    )
                    total_committed += len(new_companies)
                    # Mettre à jour les IDs existants
                    existing_ids.update(new_companies['id'].tolist())
                except Exception as e:
                    # Rollback en cas d'erreur
                    if hasattr(self.db, 'connection') and self.db.connection:
                        self.db.connection.rollback()
        
        # Réinitialiser le lot d'entreprises
        self.companies_batch = []
            
        return total_committed
    
    def commit_stocks(self):
        """Enregistrer toutes les actions dans la base de données"""
        if not self.stocks_batch:
            return 0
            
        total_committed = 0
        
        for batch in self.stocks_batch:
            if batch.empty:
                continue
                
            # Vérifier que l'index est bien 'cid'
            if batch.index.name != 'cid':
                if 'cid' in batch.columns:
                    batch = batch.set_index('cid')
                else:
                    continue
            
            # Vérifier que le DataFrame n'est pas vide après le réindexage
            if batch.empty:
                continue
                
            # CRUCIAL: Appeler df_write avec index=True et index_label='cid'
            self.db.df_write(
                batch, 
                'stocks', 
                commit=True,
                index=True,
                index_label='cid'
            )
            
            total_committed += len(batch)
        
        # Réinitialiser le lot d'actions
        self.stocks_batch = []
            
        return total_committed
    
    def commit_daystocks(self):
        """Enregistrer toutes les actions quotidiennes dans la base de données"""
        if not self.daystocks_batch:
            return 0
            
        total_committed = 0
        
        for batch in self.daystocks_batch:
            if batch.empty:
                continue
                
            # Écrire dans la base de données
            self.db.df_write(batch, 'daystocks', commit=True, index=True, index_label='cid')
            total_committed += len(batch)
        
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

def load_euronext_file(file_path):
    """Charge et analyse un fichier Euronext (CSV ou Excel)"""
    try:
        # Optimisation: Chargement direct sans préanalyse inutile
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, delimiter='\t', skiprows=None, 
                            on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return None
        
        # Standardiser les noms de colonnes
        column_mapping = {
            'Symbol': 'symbol',
            'Name': 'name',
            'Last': 'last',
            'Volume': 'volume',
            'ISIN': 'isin'    # AMÉLIORATION: Ajout de l'ISIN
        }
        
        # Renommer et standardiser les colonnes
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                df = df.drop(columns=[old_col])
        
        # Vérifier la colonne symbol
        if 'symbol' not in df.columns:
            return None
            
        # AMÉLIORATION: Nettoyer les noms (supprimer préfixe SRD)
        if 'name' in df.columns:
            df['name'] = df['name'].apply(clean_company_name)
            
        # Traiter les colonnes numériques
        # Optimisation: Utiliser la vectorisation pour le nettoyage
        if 'last' in df.columns and df['last'].dtype == 'object':
            df['last'] = df['last'].str.replace(r'[^\d.]', '', regex=True)
            df['last'] = pd.to_numeric(df['last'], errors='coerce')
        
        if 'volume' not in df.columns:
            df['volume'] = 0
        elif df['volume'].dtype == 'object':
            df['volume'] = df['volume'].str.replace(r'[^\d.]', '', regex=True)
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # AMÉLIORATION: Filtrer les volumes nuls
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
            
            # Définir le marché par défaut à Paris
            df['mid'] = 6  # ID pour le marché de Paris
            
            # AMÉLIORATION: Extraire le marché du nom de fichier si possible
            market_match = re.search(r'_([A-Za-z]+)_', filename)
            if market_match:
                market_name = market_match.group(1).lower()
                if market_name == 'amsterdam':
                    df['mid'] = 5
                    log_info(f"Amsterdam market detected in {filename}")
                elif market_name == 'london':
                    df['mid'] = 2
                    log_info(f"London market detected in {filename}")
                elif market_name == 'milan':
                    df['mid'] = 3
                    log_info(f"Milan market detected in {filename}")
                elif market_name == 'madrid':
                    df['mid'] = 4
                    log_info(f"Madrid market detected in {filename}")
                elif market_name == 'brussels':
                    df['mid'] = 8
                    log_info(f"Brussels market detected in {filename}")
                elif market_name == 'xetra':
                    df['mid'] = 7
                    log_info(f"Xetra market detected in {filename}")
            
            return df
        else:
            return None
            
    except Exception as e:
        log_error(f"Error loading Euronext file {file_path}: {str(e)}")
        return None

def resolve_conflicting_values(boursorama_value, euronext_value):
    """AMÉLIORATION: Décide quelle valeur utiliser en cas de conflit"""
    # Stratégie: utiliser la moyenne si les deux sont disponibles
    if boursorama_value is not None and euronext_value is not None:
        # Si l'écart est trop important, privilégier Euronext (considéré comme plus fiable)
        if abs(boursorama_value - euronext_value) > 0.1 * euronext_value:
            return euronext_value
        return (boursorama_value + euronext_value) / 2
    
    # Sinon utiliser la valeur disponible
    return boursorama_value if boursorama_value is not None else euronext_value

def extract_boursorama_day(date_str):
    """
    Extrait la partie date (YYYY-MM-DD) d'une chaîne de date Boursorama,
    même si elle contient des underscores
    """
    try:
        # Extraire uniquement la partie YYYY-MM-DD
        if ' ' in date_str:
            day_part = date_str.split(' ')[0]
        else:
            day_part = date_str.split('_')[0]
            
        # Vérifier si c'est bien une date
        if '-' in day_part and len(day_part.split('-')) == 3:
            return day_part
        return None
    except Exception:
        return None

@timer_decorator
def process_boursorama_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Boursorama - Version ultra-optimisée mono-thread"""
    log_info(f"Processing Boursorama files from {start_date} to {end_date}")
    
    # Initialiser notre processeur
    processor = Processor(db_model)
    
    # Convertir les dates en datetime une seule fois
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Structure du répertoire Boursorama
    boursorama_path = os.path.join(HOME, "boursorama")
    
    # Optimisation 1: Préparation des années et mois à traiter
    years_to_process = range(max(start_date.year, 2019), min(end_date.year, 2024) + 1)
    
    # Filtre des années existantes
    years_to_process = [year for year in years_to_process
                        if os.path.exists(os.path.join(boursorama_path, str(year)))
                        and os.path.isdir(os.path.join(boursorama_path, str(year)))]
    
    log_info(f"Found Boursorama data for years: {years_to_process}")
    
    # Création d'un dictionnaire date -> mois pour filtrer efficacement
    month_filters = {}
    for year in years_to_process:
        if year == start_date.year and year == end_date.year:
            # Même année - utiliser les mois entre start et end
            month_filters[year] = list(range(start_date.month, end_date.month + 1))
        elif year == start_date.year:
            # Année de début - mois restants
            month_filters[year] = list(range(start_date.month, 13))
        elif year == end_date.year:
            # Année de fin - mois jusqu'à la fin
            month_filters[year] = list(range(1, end_date.month + 1))
        else:
            # Année complète
            month_filters[year] = list(range(1, 13))
    
    # Variables pour le suivi
    files_processed = 0
    companies_added = 0
    stocks_added = 0
    daystocks_added = 0
    prev_date = None
    
    # Optimisation 2: Utiliser un cache de date pour éviter les conversions répétées
    date_cache = {}
    
    # Cache simple pour les fichiers déjà traités
    processed_files_cache = set(processed_files)
    
    # Optimisation 3: Regrouper les fichiers par date pour minimiser les commits
    for year in years_to_process:
        year_dir = os.path.join(boursorama_path, str(year))
        
        log_info(f"Scanning files in year directory {year}")
        year_files = []
        
        # Optimisation 4: Utiliser scandir pour un scan plus efficace
        with os.scandir(year_dir) as entries:
            # Convertir l'itérateur en liste pour éviter d'épuiser l'itérateur
            entry_list = list(entries)
            
            # Optimisation 5: Préfiltre rapide des fichiers avant traitement complet
            for entry in entry_list:
                # Modifié: Inclure les fichiers .bz2
                if not entry.is_file():
                    continue
                
                file_path = entry.path
                
                # Si déjà traité, ignorer
                if file_path in processed_files_cache:
                    continue
                
                try:
                    # Pour les fichiers .bz2, extraire le nom de base
                    filename = entry.name
                    if filename.endswith('.bz2'):
                        base_name = filename[:-4]  # Retirer l'extension .bz2
                    else:
                        base_name = filename
                    
                    # Filtrage rapide par nom
                    parts = base_name.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    
                    # Optimisation 6: Extraire directement les informations de date sans conversion
                    date_str = parts[1]
                    
                    # Extraire la partie date (YYYY-MM-DD) seulement pour le filtrage
                    day_part = extract_boursorama_day(date_str)
                    if not day_part:
                        continue
                    
                    # Filtre par mois - extraction directe sans conversion datetime
                    if '-' in day_part:
                        month_str = day_part.split('-')[1]
                        try:
                            month = int(month_str)
                            target_months = month_filters.get(year, [])
                            if month not in target_months:
                                continue
                        except ValueError:
                            continue
                    
                    # Optimisation 7: Utiliser le cache de dates
                    if day_part in date_cache:
                        timestamp = date_cache[day_part]
                    else:
                        timestamp = pd.to_datetime(day_part)
                        date_cache[day_part] = timestamp
                        # Utiliser également la date complète comme clé dans le cache
                        date_cache[date_str] = timestamp
                    
                    # Filtre final par date complète
                    if start_date <= timestamp <= end_date:
                        year_files.append((timestamp, file_path, parts[0]))  # parts[0] = market_name
                
                except Exception as e:
                    log_error(f"Error parsing filename {entry.name}: {str(e)}")
                    continue  # Ignorer silencieusement
        
        # Optimisation 8: Tri efficace par date
        year_files.sort()
        
        # Logging optimisé - Compter les fichiers par mois
        if year_files:
            months_count = {}
            for timestamp, _, _ in year_files:
                month = timestamp.month
                months_count[month] = months_count.get(month, 0) + 1
            
            log_info(f"Found {len(year_files)} files to process in year {year}")
            log_info(f"Files distribution by month in {year}: {months_count}")
            
            # Variables de suivi pour les logs mensuels
            last_month_logged = None
            files_processed_in_current_month = 0
            
            # Traitement séquentiel optimisé
            for timestamp, file_path, market_name in year_files:
                current_month = timestamp.month
                
                # Log mensuel pour suivre la progression
                if last_month_logged != current_month:
                    if last_month_logged is not None:
                        log_info(f">>> Completed processing {files_processed_in_current_month} files for {year}-{last_month_logged:02d}")
                    
                    last_month_logged = current_month
                    files_processed_in_current_month = 0
                    log_info(f">>> Starting to process files for {year}-{current_month:02d}")
                
                # Optimisation 9: Traitement par jour - regrouper la logique
                current_day = timestamp.date()
                if prev_date is not None and current_day != prev_date:
                    processor.process_daystocks(prev_date)
                    day_committed = processor.commit_daystocks()
                    daystocks_added += day_committed
                
                # Traiter le fichier
                companies, stocks = processor.process_boursorama_file(file_path)
                
                # Optimisation 10: Mise à jour groupée des compteurs
                companies_added += companies
                stocks_added += stocks
                
                if companies > 0 or stocks > 0:
                    files_processed += 1
                    files_processed_in_current_month += 1
                
                    # Commit périodique avec seuil optimisé
                    if stocks_added >= 50000:  # Seuil augmenté pour réduire les commits
                        stocks_committed = processor.commit_stocks()
                        log_info(f">>> Progress: Committed {stocks_committed} stock records, {files_processed} files processed")
                        stocks_added = 0  # Réinitialiser le compteur après commit
                
                # Ajouter au cache des fichiers traités
                processed_files_cache.add(file_path)
                processed_files.add(file_path)
                
                # Mettre à jour la date précédente
                prev_date = current_day
            
            # Log pour le dernier mois
            if last_month_logged is not None:
                log_info(f">>> Completed processing {files_processed_in_current_month} files for {year}-{last_month_logged:02d}")
        
        # Log de fin d'année
        log_info(f"Completed processing year {year} with {files_processed} files so far")
    
    # Traitement final des actions quotidiennes
    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added += day_committed
    
    # Optimisation 11: Nettoyage et commit final en une seule étape
    log_info("Cleaning stocks data and performing final commit...")
    processor.clean_stocks()
    
    final_companies, final_stocks, final_days = processor.commit_all()
    
    log_info(f"Boursorama processing complete: {files_processed} files processed, "
             f"{final_companies} companies added, {stocks_added + final_stocks} stocks added, "
             f"{daystocks_added + final_days} daystocks added")
    
    return files_processed, final_companies, stocks_added + final_stocks, daystocks_added + final_days

@timer_decorator
def process_euronext_files(db_model, start_date=datetime(2019, 1, 1), end_date=datetime(2024, 12, 31)):
    """Traite les fichiers Euronext - version simplifiée sans multitraitement"""
    log_info(f"Processing Euronext files from {start_date} to {end_date}")
    
    # Initialiser le processeur
    processor = Processor(db_model)
    
    # Convertir les dates en datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    directory_path = os.path.join(HOME, "euronext")
    
    # Optimisation: Recherche ciblée des fichiers en fonction des dates
    files_to_process = []
    
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            file_path = os.path.join(directory_path, item)
            if os.path.isfile(file_path):
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', item)
                if date_match:
                    file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    if start_date <= file_date <= end_date:
                        files_to_process.append((file_date, file_path))
    
    # Trier par date pour un traitement chronologique
    files_to_process.sort()
    files_to_process = [f[1] for f in files_to_process]
    
    log_info(f"Total Euronext files to process: {len(files_to_process)}")
    
    # Traiter tous les fichiers séquentiellement
    files_processed = 0
    companies_added = 0
    stocks_added = 0
    daystocks_added = 0
    
    prev_date = None
    
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        
        if date_match:
            file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
            
            # Traiter les actions quotidiennes quand le jour change
            if prev_date is not None and file_date is not None and file_date.date() != prev_date:
                processor.process_daystocks(prev_date)
                day_committed = processor.commit_daystocks()
                daystocks_added += day_committed
            
            # Charger et traiter le fichier
            df = load_euronext_file(file_path)
            if df is not None and not df.empty:
                companies = processor.process_dataframe(df)
                companies_added += companies
                stocks_added += len(df)
                files_processed += 1
            
            # Mettre à jour la date précédente
            if file_date is not None:
                prev_date = file_date.date()
    
    # Traiter les actions quotidiennes restantes
    if prev_date is not None:
        processor.process_daystocks(prev_date)
        day_committed = processor.commit_daystocks()
        daystocks_added += day_committed
    
    # Nettoyer et enregistrer à la fin
    processor.clean_stocks()
    final_companies, final_stocks, final_days = processor.commit_all()
    
    # Mettre à jour les compteurs finaux
    companies_added = final_companies
    
    log_info(f"Euronext processing complete: {files_processed} files processed, "
             f"{companies_added} companies added, {stocks_added} stocks added, "
             f"{daystocks_added} daystocks added")
    
    return files_processed, companies_added, stocks_added, daystocks_added

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
    
    # Valider les changements
    db_model.commit()
    
    # 4. Optimiser la base de données (créer un index si nécessaire)
    db_model.execute("CREATE INDEX IF NOT EXISTS idx_stocks_date_cid ON stocks(date, cid)")
    db_model.execute("CREATE INDEX IF NOT EXISTS idx_companies_symbol ON companies(symbol)")
    
    # AMÉLIORATION: Ajouter un index sur l'ISIN pour les recherches rapides
    db_model.execute("CREATE INDEX IF NOT EXISTS idx_companies_isin ON companies(isin)")
    
    db_model.commit()
    
    log_info("Database cleanup complete")

#=================================================
# SECTION 5: FONCTION PRINCIPALE
#=================================================

@timer_decorator
def main():
    """Fonction ETL principale - Version simplifiée"""
    
    log_info("="*50)
    log_info("STARTING ETL PROCESS")
    log_info("="*50)
    
    try:
        # Se connecter à la base de données en utilisant le modèle TimescaleDB
        log_info("Connecting to database...")
        db = tsdb.TimescaleStockMarketModel(
            database='bourse',
            user='ricou', 
            host='db',
            password='monmdp'
        )
        
        # Réinitialiser la base de données (comme demandé par le sujet)
        log_info("Reinitializing database...")
        db._purge_database()
        db._setup_database()
        
        # Tester la connexion à la base de données avec une requête simple
        test_result = db.df_query("SELECT 1 AS test")
        if test_result is None or test_result.empty:
            log_error("Database connection test failed")
            return 1
            
        log_info("Database connection established")
        
        # Utiliser des plages de dates basées sur la disponibilité des données connues
        start_date = datetime(2020, 5, 1)
        end_date = datetime(2020, 7, 31)
        
        #log_info(f"Using date range: {start_date.date()} to {end_date.date()}")
        
        # Traiter les fichiers Boursorama - séquentiel
        boursorama_files, boursorama_companies, boursorama_stocks, boursorama_daystocks = process_boursorama_files(
            db, start_date, end_date
        )
        
        # Traiter les fichiers Euronext - séquentiel
        euronext_files, euronext_companies, euronext_stocks, euronext_daystocks = process_euronext_files(
            db, start_date, end_date
        )
        
        # Nettoyer la base de données
        clean_database(db)
        
        # Résumé final
        companies_df = db.df_query("SELECT COUNT(*) as count FROM companies")
        stocks_df = db.df_query("SELECT COUNT(*) as count FROM stocks")
        daystocks_df = db.df_query("SELECT COUNT(*) as count FROM daystocks")
        companies_with_stocks = db.df_query("SELECT COUNT(DISTINCT cid) FROM stocks")
        
        companies_count = companies_df['count'].iloc[0] if not companies_df.empty else 0
        stocks_count = stocks_df['count'].iloc[0] if not stocks_df.empty else 0
        daystocks_count = daystocks_df['count'].iloc[0] if not daystocks_df.empty else 0
        with_stocks_count = companies_with_stocks.iloc[0, 0] if not companies_with_stocks.empty else 0
        
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
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
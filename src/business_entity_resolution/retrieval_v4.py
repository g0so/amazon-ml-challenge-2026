import re
import math
import array
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    max_df_abs: int = 5000
    max_df_ratio: float = 0.05
    top_k_tokens: int = 5
    top_k_cands: int = 50
    route_exact_name: bool = False
    route_exact_address: bool = False
    route_joint: bool = False
    exact_cap: int = 50

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

class CountryIndexV4:
    def __init__(self, db_conn, country, config: RetrievalConfig, include_train_db=False):
        self.config = config
        self.country = country
        self.target_int_to_str = []
        self.target_int_to_source = array.array('B')
        
        # Inverted lists
        self.name_index = defaultdict(lambda: array.array('I'))
        self.addr_index = defaultdict(lambda: array.array('I'))
        self.exact_n_index = defaultdict(lambda: array.array('I'))
        self.exact_a_index = defaultdict(lambda: array.array('I'))
        
        # IDFs
        self.name_idf = {}
        self.addr_idf = {}
        
        self._build_index(db_conn, include_train_db)
        
    def _build_index(self, db_conn, include_train_db):
        query = "SELECT entity_id, name_raw, address_raw, name_norm, address_norm, source FROM main.records WHERE source IN (2,3) AND country=?"
        if include_train_db:
            query += " UNION ALL SELECT entity_id, name_raw, address_raw, name_norm, address_norm, source FROM train_db.records WHERE source IN (2,3) AND country=?"
            params = (self.country, self.country)
        else:
            params = (self.country,)
            
        cur = db_conn.cursor()
        cur.execute(query, params)
        
        name_df = defaultdict(int)
        addr_df = defaultdict(int)
        
        # First Pass: Document Frequencies & Mappings
        temp_rows = []
        seen_ids = set()
        
        for row in cur:
            eid = row[0]
            if eid in seen_ids: continue # Safety deduplication
            seen_ids.add(eid)
            
            tgt_int = len(self.target_int_to_str)
            self.target_int_to_str.append(eid)
            self.target_int_to_source.append(int(row[5]))
            
            ntok = tokenize(row[1])
            atok = tokenize(row[2])
            for t in ntok: name_df[t] += 1
            for t in atok: addr_df[t] += 1
                
            temp_rows.append((tgt_int, ntok, atok, row[3], row[4]))
            
        N = len(self.target_int_to_str)
        if N == 0: return
        
        limit = min(self.config.max_df_abs, N * self.config.max_df_ratio)
        self.name_idf = {t: math.log(N / df) for t, df in name_df.items() if df <= limit}
        self.addr_idf = {t: math.log(N / df) for t, df in addr_df.items() if df <= limit}
        
        # Second Pass: Populate Postings
        for tgt_int, ntok, atok, nnorm, anorm in temp_rows:
            for t in ntok:
                if t in self.name_idf: self.name_index[t].append(tgt_int)
            for t in atok:
                if t in self.addr_idf: self.addr_index[t].append(tgt_int)
            if nnorm: self.exact_n_index[nnorm].append(tgt_int)
            if anorm: self.exact_a_index[anorm].append(tgt_int)
            
    def retrieve(self, ref_row):
        rn_raw, ra_raw = str(ref_row.get('name_raw', '')), str(ref_row.get('address_raw', ''))
        rn_norm, ra_norm = str(ref_row.get('name_norm', '')), str(ref_row.get('address_norm', ''))
        
        # Query Prep
        ntok = [(self.name_idf[t], t) for t in tokenize(rn_raw) if t in self.name_idf]
        atok = [(self.addr_idf[t], t) for t in tokenize(ra_raw) if t in self.addr_idf]
        ntok.sort(key=lambda x: (-x[0], x[1]))
        atok.sort(key=lambda x: (-x[0], x[1]))
        top_n = ntok[:self.config.top_k_tokens]
        top_a = atok[:self.config.top_k_tokens]
        
        # Score Accumulation
        n_scores = defaultdict(float)
        a_scores = defaultdict(float)
        j_scores = defaultdict(float)
        
        for idf, t in top_n:
            for tgt_int in self.name_index.get(t, []):
                n_scores[tgt_int] += idf
                if self.config.route_joint: j_scores[tgt_int] += idf
        for idf, t in top_a:
            for tgt_int in self.addr_index.get(t, []):
                a_scores[tgt_int] += idf
                if self.config.route_joint: j_scores[tgt_int] += idf
                
        cands_int = set()
        
        # 1. Baseline Token Routes
        if not self.config.route_joint:
            n_ranked = sorted(n_scores.items(), key=lambda x: (-x[1], self.target_int_to_str[x[0]]))
            a_ranked = sorted(a_scores.items(), key=lambda x: (-x[1], self.target_int_to_str[x[0]]))
            cands_int.update(tgt for tgt, _ in n_ranked[:self.config.top_k_cands])
            cands_int.update(tgt for tgt, _ in a_ranked[:self.config.top_k_cands])
        else:
            j_ranked = sorted(j_scores.items(), key=lambda x: (-x[1], self.target_int_to_str[x[0]]))
            cands_int.update(tgt for tgt, _ in j_ranked[:self.config.top_k_cands])
            
        # 2. Baseline Exact Union (AND)
        if rn_norm and ra_norm:
            ex_n = set(self.exact_n_index.get(rn_norm, []))
            ex_a = set(self.exact_a_index.get(ra_norm, []))
            cands_int.update(ex_n & ex_a)
            
        # 3. Ablation A: Independent Exact Routes with Cross-Field Capping
        if self.config.route_exact_name and rn_norm:
            ex_n = self.exact_n_index.get(rn_norm, [])
            if len(ex_n) <= self.config.exact_cap:
                cands_int.update(ex_n)
            else:
                ex_n_ranked = sorted([(tgt, a_scores.get(tgt, 0.0)) for tgt in ex_n], key=lambda x: (-x[1], self.target_int_to_str[x[0]]))
                cands_int.update(tgt for tgt, _ in ex_n_ranked[:self.config.exact_cap])
                
        if self.config.route_exact_address and ra_norm:
            ex_a = self.exact_a_index.get(ra_norm, [])
            if len(ex_a) <= self.config.exact_cap:
                cands_int.update(ex_a)
            else:
                ex_a_ranked = sorted([(tgt, n_scores.get(tgt, 0.0)) for tgt in ex_a], key=lambda x: (-x[1], self.target_int_to_str[x[0]]))
                cands_int.update(tgt for tgt, _ in ex_a_ranked[:self.config.exact_cap])
                
        # Return mapped strings and the starvation metrics (S2 vs S3 composition)
        return [self.target_int_to_str[tgt] for tgt in cands_int], [self.target_int_to_source[tgt] for tgt in cands_int]
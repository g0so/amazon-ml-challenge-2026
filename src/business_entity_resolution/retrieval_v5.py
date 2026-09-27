import re
import math
import array
import heapq
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    max_df_abs: int = 20000
    max_df_ratio: float = 0.05
    top_k_tokens: int = 5
    top_k_cands: int = 100
    route_exact_name: bool = True
    route_exact_address: bool = True
    route_joint: bool = True     # Replaces independent top-K routes when True
    exact_cap: int = 50
    union_baseline: bool = False # Preserves DF5k/K50 independent baseline candidates

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

class CountryIndexV5:
    def __init__(self, db_conn, country, config: RetrievalConfig, include_train_db=False):
        self.config = config
        self.country = country
        self.target_int_to_str = []
        
        self.name_index = defaultdict(lambda: array.array('I'))
        self.addr_index = defaultdict(lambda: array.array('I'))
        self.exact_n_index = defaultdict(lambda: array.array('I'))
        self.exact_a_index = defaultdict(lambda: array.array('I'))
        
        self.name_idf = {}
        self.addr_idf = {}
        self.base_name_idf = {}
        self.base_addr_idf = {}
        
        self._build_index(db_conn, include_train_db)
        
    def _build_index(self, db_conn, include_train_db):
        query = "SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM main.records WHERE source IN (2,3) AND country=?"
        if include_train_db:
            query += " UNION ALL SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM train_db.records WHERE source IN (2,3) AND country=?"
            params = (self.country, self.country)
        else: params = (self.country,)
            
        cur = db_conn.cursor()
        
        # --- PASS 1: DF Computation & String Mapping (Removes Memory-Heavy temp_rows) ---
        name_df = defaultdict(int)
        addr_df = defaultdict(int)
        seen_ids = set()
        
        cur.execute(query, params)
        for row in cur:
            eid = row[0]
            if eid in seen_ids: continue
            seen_ids.add(eid)
            self.target_int_to_str.append(eid)
            
            for t in tokenize(row[1]): name_df[t] += 1
            for t in tokenize(row[2]): addr_df[t] += 1
                
        N = len(self.target_int_to_str)
        if N == 0: return
        
        limit_main = min(self.config.max_df_abs, N * self.config.max_df_ratio)
        self.name_idf = {t: math.log(N / df) for t, df in name_df.items() if df <= limit_main}
        self.addr_idf = {t: math.log(N / df) for t, df in addr_df.items() if df <= limit_main}
        
        if self.config.union_baseline:
            limit_base = min(5000, N * 0.05)
            self.base_name_idf = {t: self.name_idf[t] for t, df in name_df.items() if df <= limit_base}
            self.base_addr_idf = {t: self.addr_idf[t] for t, df in addr_df.items() if df <= limit_base}
            
        del name_df
        del addr_df # Aggressive memory release before Pass 2
        
        # --- PASS 2: Postings ---
        cur.execute(query, params)
        seen_ids.clear()
        tgt_int = 0
        for row in cur:
            eid = row[0]
            if eid in seen_ids: continue
            seen_ids.add(eid)
            
            nnorm, anorm = row[3], row[4]
            for t in tokenize(row[1]):
                if t in self.name_idf: self.name_index[t].append(tgt_int)
            for t in tokenize(row[2]):
                if t in self.addr_idf: self.addr_index[t].append(tgt_int)
            if nnorm: self.exact_n_index[nnorm].append(tgt_int)
            if anorm: self.exact_a_index[anorm].append(tgt_int)
            tgt_int += 1
            
        del seen_ids

    def _get_top_k(self, score_dict, k):
        """Preserves exact sorted() tie-break parity via nsmallest while avoiding full array sort."""
        if not score_dict: return []
        return heapq.nsmallest(k, score_dict.keys(), key=lambda tgt: (-score_dict[tgt], self.target_int_to_str[tgt]))

    def retrieve(self, ref_row):
        rn_raw, ra_raw = str(ref_row.get('name_raw', '')), str(ref_row.get('address_raw', ''))
        rn_norm, ra_norm = str(ref_row.get('name_norm', '')), str(ref_row.get('address_norm', ''))
        cands_int = set()
        
        # --- Baseline Candidate Union ---
        if self.config.union_baseline:
            b_ntok = [(self.base_name_idf[t], t) for t in tokenize(rn_raw) if t in self.base_name_idf]
            b_atok = [(self.base_addr_idf[t], t) for t in tokenize(ra_raw) if t in self.base_addr_idf]
            b_ntok.sort(key=lambda x: (-x[0], x[1]))
            b_atok.sort(key=lambda x: (-x[0], x[1]))
            b_n_scores, b_a_scores = defaultdict(float), defaultdict(float)
            for idf, t in b_ntok[:5]:
                for tgt in self.name_index.get(t, []): b_n_scores[tgt] += idf
            for idf, t in b_atok[:5]:
                for tgt in self.addr_index.get(t, []): b_a_scores[tgt] += idf
            cands_int.update(self._get_top_k(b_n_scores, 50))
            cands_int.update(self._get_top_k(b_a_scores, 50))

        # --- Main Routes ---
        ntok = [(self.name_idf[t], t) for t in tokenize(rn_raw) if t in self.name_idf]
        atok = [(self.addr_idf[t], t) for t in tokenize(ra_raw) if t in self.addr_idf]
        ntok.sort(key=lambda x: (-x[0], x[1]))
        atok.sort(key=lambda x: (-x[0], x[1]))
        
        n_scores, a_scores, j_scores = defaultdict(float), defaultdict(float), defaultdict(float)
        
        for idf, t in ntok[:self.config.top_k_tokens]:
            for tgt in self.name_index.get(t, []):
                n_scores[tgt] += idf
                if self.config.route_joint: j_scores[tgt] += idf
        for idf, t in atok[:self.config.top_k_tokens]:
            for tgt in self.addr_index.get(t, []):
                a_scores[tgt] += idf
                if self.config.route_joint: j_scores[tgt] += idf
                
        if self.config.route_joint:
            cands_int.update(self._get_top_k(j_scores, self.config.top_k_cands))
        else: # Independent Route
            cands_int.update(self._get_top_k(n_scores, self.config.top_k_cands))
            cands_int.update(self._get_top_k(a_scores, self.config.top_k_cands))
            
        # --- Exact Cross-Field Buckets ---
        if rn_norm and ra_norm: cands_int.update(set(self.exact_n_index.get(rn_norm, [])) & set(self.exact_a_index.get(ra_norm, [])))
            
        if self.config.route_exact_name and rn_norm:
            ex_n = self.exact_n_index.get(rn_norm, [])
            if len(ex_n) <= self.config.exact_cap: cands_int.update(ex_n)
            else: cands_int.update(self._get_top_k({tgt: a_scores.get(tgt, 0.0) for tgt in ex_n}, self.config.exact_cap))
                
        if self.config.route_exact_address and ra_norm:
            ex_a = self.exact_a_index.get(ra_norm, [])
            if len(ex_a) <= self.config.exact_cap: cands_int.update(ex_a)
            else: cands_int.update(self._get_top_k({tgt: n_scores.get(tgt, 0.0) for tgt in ex_a}, self.config.exact_cap))
                
        # Return exact deterministic order
        return [self.target_int_to_str[tgt] for tgt in sorted(list(cands_int), key=lambda x: self.target_int_to_str[x])]
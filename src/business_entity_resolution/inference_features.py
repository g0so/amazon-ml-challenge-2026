"""The trained V3 feature semantics, with bounded per-record preprocessing cache."""
import difflib
import re
from functools import lru_cache

FEATURE_NAMES = ['name_jaccard','address_jaccard','name_exact','address_exact','name_missing','address_missing',
'name_char','address_char','name_containment','address_containment','number_overlap','number_disjoint',
'address_numbers_both','address_numbers_one','address_numbers_none','address_numbers_equal',
'address_numbers_partial','address_numbers_unshared_ref','address_numbers_unshared_target',
'name_jaccard_x_address_missing','name_char_x_address_missing','name_containment_x_address_missing']

@lru_cache(maxsize=20000)
def prepared(values):
    n,a,nn,an=values
    return (set(re.findall(r'[^\W_]+',n.lower())),set(re.findall(r'[^\W_]+',a.lower())),
            set(re.findall(r'\d+',n+' '+a)),set(re.findall(r'\d+',a)),not n.strip(),not a.strip())

def values(row):
    return tuple(str(row[k] or '') for k in ('name_raw','address_raw','name_norm','address_norm'))

def features(r,t):
    rn,ra,rnn,ran=r; tn,ta,tnn,tan=t
    nr,ar,nums_r,anums_r,nmr,amr=prepared(r)
    nt,at,nums_t,anums_t,nmt,amt=prepared(t)
    nm=float(nmr or nmt); am=float(amr or amt)
    nj=len(nr&nt)/len(nr|nt) if nr|nt and not nm else 0.
    aj=len(ar&at)/len(ar|at) if ar|at and not am else 0.
    ne=float(bool(rnn and tnn and rnn==tnn and not nm))
    ae=float(bool(ran and tan and ran==tan and not am))
    nc=(1. if rnn==tnn else difflib.SequenceMatcher(None,rnn,tnn).ratio()) if not nm else 0.
    ac=(1. if ran==tan else difflib.SequenceMatcher(None,ran,tan).ratio()) if not am else 0.
    nct=float(bool(not nm and nr and nt and (nr<=nt or nt<=nr)))
    act=float(bool(not am and ar and at and (ar<=at or at<=ar)))
    nag=float(bool(nums_r and nums_t and nums_r&nums_t))
    nco=float(bool(nums_r and nums_t and not nums_r&nums_t))
    both=float(bool(anums_r and anums_t))
    one=float(bool(anums_r)^bool(anums_t)); neither=float(not anums_r and not anums_t)
    eq=float(bool(both and anums_r==anums_t))
    part=float(bool(both and anums_r&anums_t and anums_r!=anums_t))
    ur=float(bool(both and anums_r-anums_t));ut=float(bool(both and anums_t-anums_r))
    return [nj,aj,ne,ae,nm,am,nc,ac,nct,act,nag,nco,both,one,neither,eq,part,ur,ut,nj*am,nc*am,nct*am]

def feature_job(job):
    """Top-level function for spawn workers: no target indexes copied to workers."""
    r,targets=job
    return [features(r,t) for t in targets]

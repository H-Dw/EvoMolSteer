"""Finite legal, fixed-count exchanges; no independent-atom attribution."""
import itertools
import numpy as np
from evomolsteer.nc_guidance.mining import sites


def contrasts(table):
    native=table[table.is_original].iloc[0]
    lookup={frozenset(sites(r.N_sites)):float(r.pic50_target) for r in table.itertuples()}
    original=frozenset(sites(native.N_sites));eligible=set(sites(native.eligible_sites))
    swaps=[]
    for i in sorted(original):
        for j in sorted(eligible-original):
            changed=(original-{i})|{j}
            if changed in lookup:
                swaps.append({'from_N':i,'to_N':j,'delta_score':lookup[changed]-lookup[original]})
    interactions=[]
    for a,b in itertools.combinations(swaps,2):
        if len({a['from_N'],a['to_N'],b['from_N'],b['to_N']})!=4:continue
        both=(original-{a['from_N'],b['from_N']})|{a['to_N'],b['to_N']}
        if both not in lookup:continue
        interactions.append({'swap_a':[a['from_N'],a['to_N']], 'swap_b':[b['from_N'],b['to_N']],
            'difference_of_differences':lookup[both]-lookup[original]-a['delta_score']-b['delta_score']})
    return {'kind':'fixed_geometry_legal_layouts','native_score':lookup[original],
        'n_measured_layouts':len(lookup),'exhaustive':bool(native.exhaustive),
        'native_N_sites':sorted(original),'native_minus_alternate_mean':
        float(lookup[original]-np.mean([v for k,v in lookup.items() if k!=original])) if len(lookup)>1 else None,
        'single_exchanges':swaps,'n_identifiable_interactions':len(interactions),
        'largest_signed_interactions':sorted(interactions,key=lambda x:abs(x['difference_of_differences']),reverse=True)[:12],
        'limits':'Dependent finite source inputs; not atom energies, generated outcomes, or coordinate intervention effects.'}

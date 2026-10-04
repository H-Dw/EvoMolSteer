"""Trace survivors of the LAST OBSERVED SELECTION, without final-generation data."""
import numpy as np
import pandas as pd


def window_copy_counts(groups):
    """groups: sorted event frames for one representation/arm/independent batch.

    At the last event D is the realized offspring count. Back-propagating D
    through the next event's parent edges counts copies still present just
    after the last selection. This is retrospective ancestry, not fitness.
    """
    result = [None] * len(groups)
    result[-1] = groups[-1].offspring_count.to_numpy(dtype=np.int64)
    for i in range(len(groups)-2,-1,-1):
        index = pd.Index(groups[i].node_id)
        parent = index.get_indexer(groups[i+1].parent_node_id)
        if (parent<0).any():
            raise ValueError('Window ancestry requires every intermediate parent event')
        result[i] = np.bincount(parent,weights=result[i+1],minlength=len(index)).astype(np.int64)
    if any(x.sum()!=len(g) for x,g in zip(result,groups)):
        raise ValueError('Window-end ancestry copy mass is not conserved')
    return result


def diagnostics(g, copies):
    weights = copies / copies.sum()
    score=g.pic50_on.to_numpy(float)
    return {'arm':g.arm.iloc[0], 'batch':int(g.batch.iloc[0]),
        'step':int(g.step.iloc[0]),'time':float(g.score_time.iloc[0]),
        'population':len(g),'retained_ancestor_candidates':int((copies>0).sum()),
        'extinct_candidates':int((copies==0).sum()),
        'ancestor_copy_ess':float(1/(weights@weights)),
        'current_root_count':int(g.root_id.nunique()),
        'window_end_root_count':int(g.loc[copies>0,'root_id'].nunique()),
        'current_selection_ess':float(1/np.square(g.probability).sum()),
        'model_pic50_population':float(score.mean()),
        'model_pic50_expected_selected':float(g.probability.to_numpy()@score),
        'model_pic50_window_ancestors':float(weights@score)}

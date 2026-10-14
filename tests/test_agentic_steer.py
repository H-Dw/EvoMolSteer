import ast
import numpy as np
import pytest

from evomolsteer.agentic_steer.native import instrument_native,slice_tree
from evomolsteer.agentic_steer.panel import trial_seed,candidate_indices
from evomolsteer.agentic_steer.rules import evaluate_rule


def test_native_instrumentation_preserves_loop_and_replaces_initialization():
    source='''def _generate(self):
    curr = "fresh"
    times = "fresh"
    step_sizes = [1, 1, 1, 1]
    for i, step_size in enumerate(step_sizes):
        curr += step_size
    return curr
'''
    calls=[]
    class Suffix:
        start_step=2
        def restore(self):return None,10,None,None
        def before(self,i,curr,times):calls.append(('before',i,curr))
        def after(self,i,curr,times):calls.append(('after',i,curr))
    class Model:_agentic_suffix=Suffix()
    scope={};exec(instrument_native(source),scope)
    assert scope['_generate'](Model())==12
    assert calls==[('before',2,10),('after',2,11),('before',3,11),('after',3,12)]
    with pytest.raises(ValueError):instrument_native('def _generate(self):\n    return 1')


def test_full_state_slices_times_and_soft_channels():
    import torch
    tree={'times':[torch.arange(50)]*3,'cond':torch.rand(50,25,15),'fragment_mode':['none']*50}
    sliced=slice_tree(tree,7,50,'cpu')
    assert [x.tolist() for x in sliced['times']]==[[7],[7],[7]]
    assert torch.equal(sliced['cond'],tree['cond'][7:8])
    assert sliced['fragment_mode']==['none']


def test_rng_identity_is_reproducible_and_independent():
    assert trial_seed(42,'N001',1)==trial_seed(42,'N001',1)
    assert len({trial_seed(42,'N001',r) for r in range(4)})==4


def test_candidates_do_not_use_future_labels():
    features=np.arange(100).reshape(50,2);score=np.arange(50)
    assert len(set(candidate_indices(features,score,6)))==6
    assert candidate_indices(features,score,6)[:3]==[49,24,0]


def test_executable_rule_is_active_and_cannot_escape():
    scores,audit=evaluate_rule("def score(features):\n    return np.tanh(features['a']) - .2 * np.abs(features['b'])",{'a':np.array([0.,1.]),'b':np.array([2.,0.])})
    assert scores[1]>scores[0] and audit['perturbation_response']['a']>0
    for code in ["def score(features):\n    return __import__('os')", "def score(features):\n    return np.load('a')", "def score(features):\n    return features['x']", "def score(features):\n    return features['a'] * 0"]:
        with pytest.raises(ValueError):evaluate_rule(code,{'a':np.array([0.,1.])})

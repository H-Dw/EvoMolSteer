"""Five native twenty-row streams, replayed across hypothesis groups.

Only torch RNG is borrowed. Python/NumPy/HTTP randomness is never reset by the
GPU loop. Native Gaussian and Categorical algorithms and probability formulas
are unchanged. Probability-dependent RNG consumption is an error, not a fallback.
"""
import copy
import torch


def capture():
    return {'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def restore(state):
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def equal(a, b):
    return torch.equal(a['torch_cpu'], b['torch_cpu']) and len(a['torch_cuda']) == len(b['torch_cuda']) and all(
        torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda']))


class NativeStreams:
    def __init__(self, states, groups):
        if len(states) != 5 or groups not in (1, 4):
            raise ValueError('Five historical streams and one/four groups required')
        self.states = [{k: copy.deepcopy(s[k]) for k in ('torch_cpu', 'torch_cuda')} for s in states]
        self.groups, self.calls, self.ledger = groups, 0, []

    def draw(self, kind, row_count, callback):
        if row_count != 100*self.groups:
            raise ValueError('Random tensor does not match the frozen group layout')
        external = capture()
        blocks = [None]*(5*self.groups)
        try:
            for batch in range(5):
                entry, exit_state = self.states[batch], None
                for group in range(self.groups):
                    offset = group*100+batch*20
                    restore(entry)
                    blocks[group*5+batch] = callback(offset)
                    after = capture()
                    if exit_state is not None and not equal(after, exit_state):
                        raise RuntimeError('Native random consumption differs across hypotheses')
                    exit_state = after
                self.states[batch] = exit_state
        finally:
            restore(external)
        self.calls += 1
        self.ledger.append({'call': self.calls, 'kind': kind, 'groups': self.groups, 'logical_batches': 5})
        return torch.cat(blocks, 0)

    def categorical(self, probabilities):
        return self.draw('categorical', len(probabilities),
                         lambda offset: torch.distributions.Categorical(probabilities[offset:offset+20]).sample())

    def normal(self, name, *args, **kwargs):
        if name == 'randn_like':
            value = args[0]
            if len(args) != 1:
                raise ValueError('Unexpected native randn_like signature')
            return self.draw(name, len(value), lambda offset: torch.randn_like(value[offset:offset+20], **kwargs))
        if name != 'randn':
            raise ValueError('Unsupported native random call')
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else tuple(args)
        return self.draw(name, shape[0], lambda offset: torch.randn((20, *shape[1:]), **kwargs))

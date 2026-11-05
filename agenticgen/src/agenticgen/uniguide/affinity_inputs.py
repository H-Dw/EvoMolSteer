"""Validate fixed-frame teacher inputs and normalize copies for endpoint scoring."""
import math
import numpy as np
import torch
from .io import value_sha, sha
from .native_tools import tensor_digest, tree


def model_device(model, fallback):
    parameters=list(model.parameters()) if hasattr(model,'parameters') else []
    devices={p.device for p in parameters}
    if len(devices)>1:raise ValueError('Endpoint scorer must reside on one device')
    return next(iter(devices)) if devices else fallback.device


def validate_endpoint(output):
    coords=output['coords'];mask=output['mask']
    if coords.ndim!=3 or coords.shape[-1]!=3 or not len(coords):
        raise ValueError('Endpoint coordinates must be a nonempty B x N x 3 tensor')
    if mask.shape!=coords.shape[:2] or not bool(((mask==0)|(mask==1)).all()) or not bool((mask.sum(-1)>0).all()):
        raise ValueError('Endpoint mask must contain binary, nonempty atom counts')
    for key in ('coords','atomics','bonds','charges','hybridization'):
        if key not in output:continue
        value=output[key]
        shape=(*coords.shape[:2],coords.shape[1]) if key=='bonds' else coords.shape[:2]
        if value.ndim!=len(shape)+1 or tuple(value.shape[:-1])!=tuple(shape) or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid dense endpoint tensor: '+key)


def scoring_inputs(model, endpoint, pocket, com):
    """Match runtime.scoring: world angstrom -> fixed pocket frame, decoded chemistry.

    The native raw999 finalizer deliberately returns CPU tensors. Neither this
    boundary nor a caller's COM/pocket device implies the model's device.
    """
    validate_endpoint(endpoint)
    target=model_device(model,endpoint['coords'])
    parameters=list(model.parameters()) if hasattr(model,'parameters') else []
    dtype=next((p.dtype for p in parameters if p.is_floating_point()),endpoint['coords'].dtype)
    move=lambda x:x.detach().clone().to(device=target,dtype=dtype if x.is_floating_point() else x.dtype)
    final={k:move(v) for k,v in endpoint.items() if torch.is_tensor(v) and k!='affinity'}
    center=com.detach().clone().to(device=target,dtype=final['coords'].dtype)
    if center.shape!=(len(final['coords']),3) or not bool(torch.isfinite(center).all()):
        raise ValueError('One finite world-frame pocket center is required per endpoint')
    scale=float(model.coord_scale)
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid model coordinate scale')
    # Normalize the *copy*. Integer counts feed native atom-count embeddings.
    final['mask']=final['mask'].long()
    final['coords']=(final['coords']-center[:,None,:])/scale
    for key in ('atomics','bonds','charges','hybridization'):
        if key in final:final[key]=torch.nn.functional.one_hot(final[key].argmax(-1),final[key].shape[-1]).to(dtype)
    pocket=tree(pocket,move)
    if 'coords' in pocket and (len(pocket['coords'])!=len(final['coords']) or not bool(torch.isfinite(pocket['coords']).all())):
        raise ValueError('Pocket rows or coordinates do not match endpoint rows')
    times=[torch.full((len(final['coords']),),.9999,device=target,dtype=dtype) for _ in range(3)]
    return final,pocket,times


def audit_teacher(model,teacher,protein):
    from .absolute_coordinate import validate
    from .geometry_chemistry import condition_features
    condition=validate(teacher['condition']);features=teacher['features']
    if (condition['features_sha256']!=value_sha(features)
            or condition['atoms']!=features['atoms'] or condition['bonds']!=features['bonds']
            or condition['reference_coordinates_angstrom']!=features['coordinates_angstrom']
            or condition['controlled_atom_ids']!=sorted(features['controlled_atom_ids'])):
        raise ValueError('Teacher graph, geometry and controlled identities are not bound together')
    if sha(protein)!=condition['protein_alignment']['moving_protein_sha256']:
        raise ValueError('Teacher condition belongs to another protein frame')
    channels,_,binding=condition_features(model,condition)
    return dict(teacher_id=teacher['id'],source_id=teacher['source_id'],condition_sha256=condition['condition_sha256'],
        features_sha256=value_sha(features),protein_sha256=sha(protein),atoms=len(condition['atoms']),
        controlled_atom_ids=condition['controlled_atom_ids'],chemical_condition_sha256=binding['chemical_condition_sha256'],
        channels={k:list(v.shape) for k,v in channels.items()},
        inpainting_scope='controlled_atoms_and_induced_bonds',full_graph_scoring_does_not_expand_inpainting=True)


def audit_initial(initial,reference,condition):
    from .iterative_affinity import validate_prior
    n=len(condition['atoms']);validate_prior(initial['prior'],reference['prior'],n)
    if initial.get('population')!=50 or len(initial['stream_states'])!=5:
        raise ValueError('Fresh inference requires fifty rows and five independent seed streams')
    validate_endpoint(initial['prior'])
    if not bool((initial['prior']['mask'].sum(-1)==n).all()):
        raise ValueError('Generated slot counts differ from the current teacher heavy-atom count')
    for key in ('pocket_target','pocket_off','target_com'):
        if tensor_digest(initial[key])!=tensor_digest(reference[key]):
            raise ValueError('Fresh inference changed the frozen pocket/frame: '+key)
    center=initial['target_com']
    if center.shape!=(50,3) or not bool(torch.isfinite(center).all()) or not torch.equal(center,center[:1].expand_as(center)):
        raise ValueError('Initial teacher scoring and candidate scoring require the same fixed pocket center')
    pocket=initial['pocket_target']
    if not torch.equal(pocket['coords'],pocket['coords'][:1].expand_as(pocket['coords'])):
        raise ValueError('Initial teacher scoring and candidate scoring use different protein pockets')
    return dict(population=50,atoms=n,prior={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in initial['prior'].items() if torch.is_tensor(v)},
        pocket_sha256=tensor_digest(pocket),target_com_sha256=tensor_digest(center),
        frame='world angstrom output; (output - target_com) / coord_scale for affinity only',
        reference_pocket_and_center_unchanged=True)


def audit_original_molecule(features,mol,template):
    """Bind the baseline's complete SDF to the interaction-only sampling teacher."""
    from .geometry_condition import from_molecule
    actual=from_molecule(mol,features['controlled_atom_ids'],coordinate_frame='generation_context_angstrom')
    if any(actual[k]!=features[k] for k in ('atoms','bonds')):
        raise ValueError('Initial teacher scoring SDF and guidance graph differ')
    xyz=np.asarray(actual['coordinates_angstrom']);expected=np.asarray(features['coordinates_angstrom'])
    if xyz.shape!=expected.shape or not np.allclose(xyz,expected,rtol=0,atol=5.1e-5):
        raise ValueError('Initial teacher scoring SDF and guidance coordinates differ')
    ids=[a['id'] for a in actual['atoms']]
    if sorted(ids[i] for i in template['core_sdf_atoms'])!=sorted(features['controlled_atom_ids']):
        raise ValueError('Initial guidance identities differ from the frozen PLIP interaction site')
    if mol.GetNumHeavyAtoms()!=template['heavy_atoms'] or sha(template['protein'])!=template['protein_sha256']:
        raise ValueError('Frozen teacher atom count or protein changed')
    return dict(sdf_sha256=sha(template['sdf']),protein_sha256=sha(template['protein']),
        graph_matches=True,coordinate_max_error_angstrom=float(np.abs(xyz-expected).max()),
        plip_controlled_atoms_match=True)

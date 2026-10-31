"""One fixed-scale measurement; old calibration supplies parameters, never receipts."""
from pathlib import Path
from .io import read, sha, frozen, value_sha

MODE = "fixed_gradient_atom_scope"


def validate_fixed_reference(config, teacher, condition):
    if config.get("experiment_mode") != MODE or type(config.get("rounds")) is not int or config["rounds"] != 0:
        raise ValueError("Explicit zero-iteration fixed-scope mode is required")
    from .gradient_atom_scope import policy
    if policy(config)["mode"] != "condition_hops":
        raise ValueError("Fixed-scope experiments require an explicit 0/1/2-hop mask")
    if config.get("prefix_inpainting", {}).get("enabled", False):
        raise ValueError("This experiment uses the baseline without joint prefix")
    if config.get("windows") != [[10,25],[26,50],[51,75],[76,99]]:
        raise ValueError("Fixed-scope experiments retain the baseline four windows")
    reference = config["fixed_scale_reference"]
    baseline = Path(reference["baseline_root"]).resolve()
    path = baseline / 'calibration/domains.json'
    if sha(path) != reference["domains_file_sha256"]:
        raise ValueError("Baseline pre-calibration domains changed")
    domain = read(path)[teacher["target"]]
    if domain['condition_sha256'] != condition['condition_sha256'] or domain.get('evidence_revision') != 0:
        raise ValueError("Wrong condition or Agent-revised calibration reference")
    source = (baseline / domain['initial_source']).resolve()
    if not source.is_relative_to(baseline/'calibration'):
        raise ValueError("The reference must be pre-Agent calibration evidence")
    if sha(source/'COMPLETE.json') != reference['source_complete_sha256']:
        raise ValueError("Baseline source receipt changed")
    from .geometry_sampling import validate_receipt
    measured = validate_receipt(source)
    schedule = config['fixed_guidance_schedule']
    expected = [domain['calibrated_guidance_reference_scale']] * 4
    if schedule != expected or measured['effective_guidance_scale_schedule'] != expected or measured['attempts'] != 50:
        raise ValueError("Use the baseline pre-calibration reference unchanged in every window")
    return dict(reference, teacher_id=teacher['target'], baseline_initial_source=domain['initial_source'],
                schedule=expected, new_calibration_performed=False, old_measurements_imported=False,
                masked_strategy_is_not_claimed_precalibrated=True)


def validate_fixed_call(config, teacher, condition, schedule, stage):
    if config.get('experiment_mode') != MODE or stage != 'formal':
        raise ValueError("Actual strengths outside pre-experiments require explicit fixed-scope mode")
    validate_fixed_reference(config, teacher, condition)
    if list(schedule) != config['fixed_guidance_schedule']:
        raise ValueError("Fixed scale is immutable; no search or Agent-selected strengths")


def run(config):
    from .absolute_runtime import validate_config, require_acceptance, calibrated
    from .geometry_sampling import execute
    from .prepare import load_model
    from .quality_queue import enqueue
    import torch
    config = validate_config(config)
    require_acceptance(config)
    subjects = calibrated(config)
    if len(subjects) != 1:
        raise ValueError("One teacher per fixed-scope job")
    teacher, condition = subjects[0]['teacher'], subjects[0]['condition']
    reference = validate_fixed_reference(config, teacher, condition)
    root = Path(config['root'])
    frozen(root/'ATTEMPT_LIMITS.json', {'stage_caps':{'formal':50,'calibration':0,'r0':0}})
    frozen(root/'FIXED_SCALE_PROVENANCE.json', reference)
    model = load_model(config, teacher)
    try:
        result = execute(model, config, teacher, condition, config['fixed_guidance_schedule'],
                         root/'fixed_population', 'formal', count=50)
    finally:
        del model
        torch.cuda.empty_cache()
    enqueue(config, root/'fixed_population', teacher)
    verify_trace(root/'fixed_population'/'trace.h5', config['guidance_atom_scope']['hops'], root/'GRADIENT_SCOPE_VERIFIED.json')
    frozen(root/'SAMPLING_FINISHED.json',dict(mode=MODE,population=50,iterations=0,llm_calls=0,
                                           source='fixed_population',reference=value_sha(reference)))
    return dict(result, guidance_agent_invoked=False, iterations=0, llm_calls=0)


def verify_trace(path, hops, output):
    """Recompute saved per-step radii and native vs added motion on CPU."""
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as trace:
        masks = trace['guidance_atom_mask'][:].astype(bool)
        valid = trace['mask'][:].astype(bool)
        assignment = trace['assignment'][:]
        expected = np.zeros_like(masks)
        batch = np.arange(masks.shape[1])[:,None]
        for step in range(100):
            expected[step][batch, assignment[step]] = True
            if hops:
                labels = trace['guidance_scope_bond_classes'][step]
                adj = (labels != 0) | (labels.transpose(0,2,1) != 0)
                adj &= valid[step,:,None,:] & valid[step,:,:,None]
                for _ in range(hops):
                    expected[step] |= (adj & expected[step,:,:,None]).any(axis=1)
        if not np.array_equal(expected & valid, masks):
            raise ValueError('Saved guidance scope differs from the declared per-step graph radius')
        delta = trace['guidance_delta_A'][:]
        native, guided = trace['native_proposal'][:], trace['guided_proposal'][:]
        raw, applied = trace['unmasked_input_gradient'][:], trace['input_gradient'][:]
        if np.any(delta[~masks] != 0) or not np.array_equal(native[~masks], guided[~masks]):
            raise ValueError('Extra guidance modified an atom outside its declared scope')
        if not np.array_equal(raw[masks], applied[masks]) or np.any(applied[~masks] != 0):
            raise ValueError('The scope mask changed a selected VJP or retained an outside VJP')
        report=dict(hops=hops,steps=100,population=masks.shape[1],
                    graph_radius_recomputed_exact=True,outside_extra_increment_exact_zero=True,
                    native_motion_outside_scope_unchanged=True,selected_full_vjp_unchanged=True,
                    selected_atoms_min=int(masks.sum(-1).min()),selected_atoms_max=int(masks.sum(-1).max()),
                    raw_vjp_outside_scope_maximum=float(np.abs(raw[~masks]).max()) if (~masks).any() else 0.,
                    model_forwards_during_verification=0,llm_calls=0)
    frozen(output,report)
    return report

"""Read-only action admission shared by tools, both roles and execution preflight."""
from pathlib import Path

from . import linear_scale as scale
from .checkpoint_learning import context as checkpoint_context
from .io import read, sha


def budget_state(config, domain):
    from .linear_runtime import formal_allowance, range_allowance
    root = Path(config['root'])
    ledger = root/'attempts.json'
    rows = read(ledger) if ledger.exists() else {}
    used = sum(r['count'] for r in rows.values()
               if r['stage']=='formal' and r.get('teacher_id')==domain['teacher_id'])
    teacher_cap = 50*config['rounds']*domain['control_settings']['maximum_selected']
    return dict(remaining_formal_attempts=min(formal_allowance(config), max(0,teacher_cap-used)),
                range_evidence_allowance=range_allowance(config,domain['teacher_id']),
                attempts_sha256=sha(ledger) if ledger.exists() else None,
                limits_sha256=sha(root/'ATTEMPT_LIMITS.json'),
                scalar_confirmation_is_admission_whitelist=not scale.open_domain(domain),
                declared_grid=(sorted({v for g in domain['search_grids'] for v in g}) if 'search_grids' in domain else scale.grid(domain['control_settings'])),
                **(dict(declared_window_grids=domain['search_grids'],distance_unit='per_window_grid_index',
                    execute_all_proposed=True) if 'search_grids' in domain else {}),
                confirmation_evidence_levels=domain['confirmed_effective_levels'],
                unknown_configuration_stability='unmeasured; no safety or improvement guarantee')


def audit_actions(config, history, domain, actions, failures=(), learning=None, *, portfolio=False, evidence_ids=(), search_focus=None, stage_partition=None):
    from .linear_runtime import remaining_profile_cost
    from .path_search import enabled as path_search
    from . import stage_search
    stage_search.validate_binding(config,domain)
    state = budget_state(config,domain)
    available = state['remaining_formal_attempts']
    frontier = dict(state['range_evidence_allowance'])
    learning = learning or (checkpoint_context(config,history,[],domain) if scale.open_domain(domain) else None)
    anchors = {r['id']:r for r in history}
    seen, results = list(history), []
    total_formal, total_range, total_configurations = 0, 0, 0
    for action in actions:
        try:
            if path_search(domain):
                import jsonschema
                try: jsonschema.validate(action, scale.action_schema(anchors, domain))
                except jsonschema.ValidationError as error: raise ValueError(error.message) from error
                if not action['evidence_ids'] or set(action['evidence_ids']) - (set(anchors) | set(evidence_ids)):
                    raise ValueError('Use measured history or verified observed evidence receipts')
                if sum(a.get('id') == action['id'] for a in actions) > 1:
                    raise ValueError('Repeated proposal identity')
            resolved = scale.resolve_action(action,anchors,domain,seen,failures)
            if scale.open_domain(domain):
                target = action['optimization_target']
                if (target['incumbent_source_id'] != learning['best_checkpoint_id'] or
                    target['reviewed_path_sha256'] != learning['best_path_sha256']):
                    raise ValueError('Optimization target does not bind the current measured best decision path')
                if learning['rebase_from_best_checkpoint'] and not path_search(domain) and (
                    action['anchor_id'] != learning['best_checkpoint_id'] or
                    target['path_relation'] != 'redirect_from_reviewed_best'):
                    raise ValueError('Rollback must redirect from the reviewed measured best node')
            try:
                cost = remaining_profile_cost(config['root'],domain['teacher_id'],resolved[scale.EFFECTIVE],domain.get('guidance_box_mask'))
            except ValueError as error:
                from .analysis_tools import EvidenceIntegrityError
                raise EvidenceIntegrityError('Measured admission source changed: '+str(error)) from error
            except RuntimeError as error:
                raise ValueError(str(error)) from error
            stage = 'calibration' if resolved['task_type']=='range_evidence' else 'formal'
            allowed = ((frontier['remaining_configurations']>=1 and frontier['remaining_attempts']>=cost)
                       if stage=='calibration' else available>=cost)
            admission = dict(executable_now=allowed,stage=stage,required_new_attempts=cost,
                             reason=None if allowed else ('frontier_budget_exhausted' if stage=='calibration' else 'formal_budget_exhausted'))
            if learning:
                changes=[abs(a-b)/domain[scale.UNIT] for a,b in zip(
                    resolved[scale.EFFECTIVE],learning['best_checkpoint_actual_schedule'])]
                if 'search_grids' in domain:
                    from .box_search import changes as grid_changes
                    changes=grid_changes(resolved[scale.EFFECTIVE],learning['best_checkpoint_actual_schedule'],domain)
                admission['distance_from_global_best_ticks']=dict(total=sum(changes),maximum_window=max(changes))
            results.append(dict(action=action,status='executable_unmeasured' if allowed else 'not_executable',
                                resolved=resolved,admission=admission,observed_performance=None))
            seen.append(resolved)
            if stage=='formal':
                total_formal+=cost
                if portfolio: available-=cost
            else:
                total_range+=cost;total_configurations+=1
                if portfolio:
                    frontier['remaining_attempts']-=cost;frontier['remaining_configurations']-=1
        except ValueError as error:
            results.append(dict(action=action,status='rejected',reason=str(error),
                                admission=dict(executable_now=False)))
    all_valid = all(r['status']!='rejected' for r in results)
    batch_ok = (all_valid and total_formal<=state['remaining_formal_attempts']
                and total_range<=state['range_evidence_allowance']['remaining_attempts']
                and total_configurations<=state['range_evidence_allowance']['remaining_configurations'])
    diversity_error=None
    early_audit = None
    if portfolio and all_valid and (domain.get('box_exclusion_policy') or path_search(domain)):
        from .profile_diversity import bank, require_portfolio
        try:
            from . import stage_search
            if stage_search.enabled(domain):
                early_audit = stage_search.audit(actions,history,domain,failures,
                    state["remaining_formal_attempts"],learning,search_focus,stage_partition)
            else:
                suggestions=bank(history,domain,learning['best_checkpoint_actual_schedule'],failures)
                if actions or not path_search(domain):
                    require_portfolio(actions,domain,learning['best_checkpoint_actual_schedule'],suggestions,state['remaining_formal_attempts'])
                if path_search(domain):
                    from .path_search import audit_early
                    early_audit = audit_early(actions,history,domain,failures,state["remaining_formal_attempts"])
        except ValueError as error:
            batch_ok=False;diversity_error=str(error)
    return dict(status='complete',actions=results,budget_state=state,
                **(dict(early_direction_audit=early_audit) if path_search(domain) else {}),
                evaluation_scope='selected_portfolio' if portfolio else 'independent_alternatives',
                batch_admission=dict(executable_now=batch_ok,formal_new_attempts=total_formal,
                                     calibration_new_attempts=total_range,frontier_configurations=total_configurations,
                                     **({'diversity_error':diversity_error} if diversity_error else {})),
                new_molecular_attempts=0)


def require_executable(report, *, portfolio):
    bad = [r for r in report['actions'] if not r['admission']['executable_now']]
    if bad or (portfolio and not report['batch_admission']['executable_now']):
        raise ValueError('Action is not executable with the current measured context/budget: '+
                         str([r.get('reason',r['admission'].get('reason')) for r in bad]
                             or report['batch_admission']))

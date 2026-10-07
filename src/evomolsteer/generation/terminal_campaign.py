"""Registered descendant-credit target comparisons within the same new30 budget."""
from .regional_campaign import propose_regional
from .affinity_campaign import select_parent


def propose_terminal(number, rows, programs, prior):
    if number in (17, 19, 24):
        parent = select_parent([r for r in rows if r['round'] <= 26], secondary=True)
        plan = dict(parent_round=parent['round'], template_round=17 if number != 17 else None,
            terminal_credit=True, reward_view='endpoint_pointcloud', time_ramp_power=0.,
            native_rms_ratio=.3, teacher_neighbors=4, teacher_score_beta=2., teacher_endpoint_temperature_A2=4.,
            action='terminal_credit_target_revision', global_parent_retained=True,
            reason='Compare intact ancestral endpoint coordinates with offline mean descendant credit; extinction remains missing and global affinity winner stays available.')
        if number == 19:
            plan.update(teacher_score_beta=0., action='terminal_credit_prior_counterfactual',
                reason='Retain surviving ancestor geometry but remove its conditional score weighting; no online head or causal survival claim.')
        if number == 24:
            plan.update(native_rms_ratio=.18, action='terminal_credit_dose_counterfactual',
                reason='Test lower dose on the collapsed ancestral library; avoid escalating a potentially biased coordinate target.')
        return plan
    return propose_regional(number, rows, programs, prior)

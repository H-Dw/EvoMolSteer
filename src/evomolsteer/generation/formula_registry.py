"""Shared reward capabilities, separate from experimental role advice."""
from pathlib import Path
from ..io import digest


def formula_registry(root):
    root=Path(root)
    sources=['affinity_geometry_reward.py','endpoint_reward.py','sparse_reward.py',
             'persistent_reward.py','regional_point_reward.py','endpoint_guidance.py']
    return {
        'schema_version':'coordinate-reward-capabilities-1.0',
        'purpose':'Execution definitions shared by every Designer condition, not preferred architectures.',
        'symbols':{'X_t':'current coordinates','Y_t':'native FLOWR endpoint forecast',
            'Phi':'registered coordinate observables divided by recorded feature scale',
            'V':'recorded scaled feature variance','s_m':'recorded teacher score',
            'theta':'scalar parameters supplied in program registry',
            'rho_delta(q)':'delta^2 * (sqrt(1 + q/delta^2) - 1)'},
        'families':{
            'endpoint_pointcloud':{
                'source':'endpoint_reward.py + affinity_geometry_reward.py',
                'correspondence':'Hungarian squared-distance assignment between detached Y_t and each teacher endpoint; choose K nearest by mean assignment cost C_m.',
                'prior':'pi_m = softmax(-C_m/T + beta*(s_m-mean(s)) + optional teacher_base_log_weight_m); detached.',
                'formula':'q_m = mean_i ||Y_t,i - teacher_m,assignment(i)||^2; R = tau*logsumexp_m(log(pi_m)-rho_delta(q_m)/tau).'},
            'endpoint_direction':{
                'source':'sparse_reward.py or affinity_geometry_reward.py',
                'formula':'R = sum_f Phi_f*d_f. Node contrast d = clip(mean_batches((high-low)/V),-3,3)*block_normalized_feature_weights. Recorded empirical/approved polynomial direction uses effect_f(t)/mean(V_f), clipped then weighted.',
                'scope':'Only registered fields/functions and their supplied fidelity certificates.'},
            'endpoint_supported_attractor':{
                'source':'persistent_reward.py',
                'formula':'q_m = sum_f (Phi_f - teacher_Phi_m,f)^2 * w_f(t)/mean(V_f); attraction = tau*logsumexp(log_softmax(beta*(s-mean(s)))-rho_delta(q)/tau). R = attraction - h*phase^p*rho_delta(sum_f (Phi_f-Phi_previous_detached_f-observed_increment_f)^2*w_history_f/mean(V_f)).',
                'scope':'History penalty only when immediate previous supported endpoint exists. Recorded salience, when active, multiplies weights by clip(abs(effect),0.1,2).'},
            'endpoint_regional_pointcloud':{
                'source':'regional_point_reward.py',
                'prior':'Same unweighted detached assignment and nearest-teacher prior as pointcloud, without optional base log weight.',
                'formula':'a_i = background + sum_r w_r*node_salience_r(t)*exp(-||Y_anchor,i-landmark_r||^2/(2*radius^2)); detach a. q_m = sum_i a_i*||Y_t,i-teacher_m,assignment(i)||^2/sum_i a_i. R = tau*logsumexp(log(pi_m)-rho_delta(q_m)/tau).'}},
        'derivative_and_dose':'g_t = J_FLOWR_endpoint(X_t)^T * grad_Y R. Use existing native forward graph, no affinity-head gradient or additional production forward. Relative RMS dose and bounded backtracking are supplied program controls; ramp factor = (0.1+0.9*phase)^time_ramp_power.',
        'support':'Exact reference clocks and learned window supplied by data; native continuation outside support.',
        'choice_contract':'Registered historical programs may be retained or changed within declared scalar bounds, or deferred; no architecture or parameter is required to win.',
        'source_sha256':{name:digest(root/'src/evomolsteer/generation'/name) for name in sources if (root/'src/evomolsteer/generation'/name).exists()}}

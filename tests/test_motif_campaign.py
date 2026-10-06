from evomolsteer.generation.motif_campaign import pareto_front,choose,proposal,AXES


def row(n,head=0,physical=0):
    r={k:physical for k in AXES};r.update(round=n,all_head_change_vs_native=head,
        valid_rate_change=0,PB_rate_change=0,energy_coverage_change=0)
    return r


def test_pareto_keeps_tradeoff_and_drops_dominated_program():
    a,b,c=row(1,.1,.1),row(2,.2,-.1),row(3,0,-.2)
    assert [v['round'] for v in pareto_front([a,b,c])]==[1,2]
    assert choose([a,b,c])['round']==1
    assert choose([a,b,c],'affinity')['round']==2


def test_predeclared_architectures_are_distinct_and_no_graph_gate():
    plans=[proposal(i,'unused') for i in range(1,9)]
    assert len({(p['channel'],p['reward_view'],p['motif_components']) for p in plans})==7
    assert all('graph' not in k for p in plans for k in p)


def test_boundary_target_plans_and_count_threshold_roundoff():
    for n in (9,10):assert proposal(n,'unused')['target_definition']=='boundary_survival'
    a,b=row(1,.01,.1),row(2,.1,-.1)
    a['PB_rate_change']=.96-.98
    assert choose([a,b])['round']==1

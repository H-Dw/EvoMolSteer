import argparse
from evomolsteer.continuous.motif_agents import export,import_response,compile_design
p=argparse.ArgumentParser();p.add_argument('--action',choices=('export','import','api','compile'),required=True)
for k in ('mining','output','request','response','reference'):p.add_argument('--'+k)
p.add_argument('--role',choices=('Analyst','Designer'));p.add_argument('--round',type=int)
a=p.parse_args()
if a.action=='export':export(a.mining,a.role,a.output)
elif a.action=='import':import_response(a.request,a.response,a.output)
elif a.action=='compile':compile_design(a.response,a.reference,a.output,a.round)
else:
    from evomolsteer.agents import call_api
    call_api(a.request,a.output)

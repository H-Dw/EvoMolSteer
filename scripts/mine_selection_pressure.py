import argparse
from evomolsteer.continuous.selection_pressure import mine,augment_reference
from evomolsteer.io import write_json

p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--augmented-reference',required=True)
a=p.parse_args();print(mine(a.dataset,a.campaign,a.reference,a.output))
write_json(a.output+'/niche_audit.json',augment_reference(a.reference,a.augmented_reference))

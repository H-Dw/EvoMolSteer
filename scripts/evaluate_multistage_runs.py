import argparse
from evomolsteer.generation.multistage_evaluation import calibrate,evaluate

p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
p.add_argument('--calibrate',action='store_true');a=p.parse_args()
print(calibrate(a.campaign,a.output) if a.calibrate else evaluate(a.campaign,a.output).to_string(index=False))

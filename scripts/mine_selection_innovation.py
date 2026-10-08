"""Deterministic compact selection-innovation tool, usable by either Agent backend."""
import argparse
import json
from evomolsteer.continuous.selection_innovation import mine

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--neighbours',type=int,default=6)
    args=parser.parse_args()
    result=mine(args.dataset,args.campaign,args.reference,args.output,args.seed,args.neighbours)
    print(json.dumps({'schema_version':result['schema_version'],'events':result['events'],
                      'teacher_records':result['teacher_records'],'augmented_reference_sha256':result['augmented_reference_sha256'],
                      'maximum_teacher_match_RMS_A':result['maximum_teacher_match_RMS_A']},sort_keys=True))

if __name__=='__main__':main()

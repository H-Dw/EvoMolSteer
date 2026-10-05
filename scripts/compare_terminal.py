import argparse
from evomolsteer.generation.terminal_comparison import compare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(compare(a.manifest, a.output)['decisions'])

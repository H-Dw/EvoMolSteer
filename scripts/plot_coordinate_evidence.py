import argparse
from evomolsteer.continuous.coordinate_plot import plot

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('mining','features','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();plot(a.mining,a.features.split(','),a.output)

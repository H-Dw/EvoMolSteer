"""Use the audited serial campaign; new calibration includes box Selection."""
import importlib.util
import sys
from pathlib import Path


def load_runner():
    directory=Path(__file__).resolve().parents[1]/'20261031_guidance_comparison5'
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('box_serial_runner',directory/'run_suite.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    formal_sources=common.quality_sources
    def quality_sources(config):
        root=Path(config['root']);sources=set(formal_sources(config))
        for path in (root/'box_exclusion').glob('*/profiles/*/population_50/COMPLETE.json'):
            row=common.read(path.parent/'aggregate.json')
            if row.get('complete') and row.get('attempts')==50:sources.add(row['id'])
        return sorted(sources)
    common.quality_sources=quality_sources
    return common


if __name__ == '__main__':
    load_runner().main()

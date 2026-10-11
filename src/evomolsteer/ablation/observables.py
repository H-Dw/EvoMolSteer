"""Source-bound observable semantics shared by every scientific condition."""
import gzip
import json
from pathlib import Path

from evomolsteer.continuous.affinity_geometry import names, PAIR_RADII
from evomolsteer.continuous.m2_skill_api import REF
from evomolsteer.io import digest, read_json, write_json
from .evidence import REGIONAL


def build(repo,study):
    repo,study=Path(repo),Path(study)
    reference=json.loads(gzip.decompress((repo/REF).read_bytes()))
    prior=read_json(repo/REGIONAL/'region_prior.json')
    region_source=repo/'src/evomolsteer/continuous/endpoint_regions.py'
    geometry_source=repo/'src/evomolsteer/continuous/affinity_geometry.py'
    if prior['source_code_sha256']!=digest(region_source):
        raise ValueError('Historical regional extractor does not match supplied definitions')
    if reference['features']!=names(reference['landmarks_A']):
        raise ValueError('Geometry feature vocabulary does not match supplied definitions')
    definitions={
        'schema_version':'coordinate-observable-definitions-1.0',
        'source_code_sha256':{
            region_source.relative_to(repo).as_posix():digest(region_source),
            geometry_source.relative_to(repo).as_posix():digest(geometry_source)},
        'semantics_source':'Exact coordinate extraction formulas; common data dictionary, not a research instruction or a favorable direction',
        'coordinates':'x_i and landmarks l_r in the common receptor frame, Angstrom; c=mean_i(x_i); N is active slot count',
        'global_centroid':{'formula':'c - supplied origin_A','units':'A'},
        'shape':{'formula':'C=mean_i[(x_i-c)(x_i-c)^T]; diagonal entries C_aa; off-diagonal entries sqrt(2)*C_ab','units':'A^2'},
        'pair_kernel':{'formula':'mean over ordered i != j of exp(-0.5*((||x_i-x_j||-a)/sigma)^2)',
            'a_A':list(PAIR_RADII),'sigma_A':.75,'units':'dimensionless',
            'meaning':'Radial pair-distance shell density; not a bond label or interaction energy'},
        'landmark_softmin':{'formula':'-T*(logsumexp_i(-||x_i-l_r||/T)-log(N))',
            'T_A':.5,'units':'A','meaning':'Normalized soft minimum distance; lower means geometrically closer, without establishing a quality benefit'},
        'landmark_occupancy3_and_occupancy5':{
            'formula':'mean_i exp(-0.5*((||x_i-l_r||-a)/sigma)^2)','a_A':[3.,5.],'sigma_A':1.,'units':'dimensionless',
            'meaning':'Radial shell density peaking at a; NOT the number of atoms within radius a, and not monotone in distance'},
        'regional_fields':{
            'radius_A':prior['radius_A'],
            'weights':'w_i=exp(-||x_i-l_r||^2/(2*radius_A^2))',
            'soft_mass':{'formula':'sum_i w_i/N','units':'dimensionless'},
            'local_dx_dy_dz':{'formula':'mu=sum_i[w_i*(x_i-l_r)]/sum_i w_i','units':'A'},
            'local_cxx_cyy_czz_cxy_cxz_cyz':{
                'formula':'sum_i[w_i*((x_i-l_r)-mu)*((x_i-l_r)-mu)^T]/sum_i w_i; raw entries without sqrt(2) scaling',
                'units':'A^2'}},
        'contrast':'Effect_z is a label-bound favorable-minus-comparison contrast divided by the original feature scale. A sign on one observable cannot be compared to another as a shared quality axis without its definition.',
        'scope':'Endpoint forecast descriptors, not atom velocities, neural attention, physical energies or causal affinity gradients. Coordinate covariance is not by itself an inferred chemical mechanism.'}
    path=study/'evidence/observable_definitions.json'
    if path.exists() and read_json(path)!=definitions:
        raise ValueError('Refuse to replace frozen observable definitions')
    if not path.exists():write_json(path,definitions)
    return path

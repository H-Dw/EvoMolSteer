"""Measured spatial-motif distributions, with free atom/bond generation."""
import gzip,json,math
from pathlib import Path
import numpy as np
import torch
from ..continuous.spatial_motifs import numpy_motifs,torch_motifs
from ..io import read_json,digest
from .prototypes import write_json
from ..storage.trajectory import TrajectoryPackage
from .coordinate_shape import time_weights,dimensionless_moments
from .coordinate_reward import CoordinateMixtureReward


def build(dataset,campaign,mining,output,channels=('all',),target='instantaneous'):
    root,mining,output=map(Path,(dataset,mining,output))
    if output.exists():raise FileExistsError(output)
    m,cat=read_json(mining/'manifest.json'),read_json(mining/'feature_catalog.json')
    if m['feature_family']!='motif' or m['control_representation']!='proposal' or m['spatial_anchor']!='endpoint':raise ValueError('Matched motif evidence required')
    if not channels or set(channels)-{'all','NOS','NOS_C'}:raise ValueError('Unknown motif channels')
    if target not in ('instantaneous','boundary_survival'):raise ValueError('Unknown lineage target')
    source=root/'results'/campaign;cfg=read_json(source/'config.json');frames={};sources=[]
    for b in m['splits']['discovery']:
        path=source/'single'/f'batch_{b:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{b:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as z:
            times=np.round(z.read('score_time')[:,0].astype(float),6)
            survival={}
            if target=='boundary_survival':
                from ..continuous.coordinate_mining import boundary_descendants
                # Survival is at the actual control boundary, not the .51
                # proposal of the last scored .5 event or a terminal t=1 label.
                inside,copies=boundary_descendants(z.read('resampled'),z.read('state_time'),z.read('selected_indices'),m['window'])
                survival={int(i):copies[k] for k,i in enumerate(inside)}
            for i in np.flatnonzero(z.read('resampled')):
                t=float(times[i]);x=z.read('proposal_coords',int(i)).astype(float)*cfg['coord_scale']+com
                anchor=z.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                v,names,_=numpy_motifs(x,z.read('predicted_atomics',int(i)),z.read('mask',int(i)),cat,anchor,cat['spatial_width_A'],channels)
                valid=np.isfinite(v).all(1)
                weights=(survival.get(int(i),np.ones(len(v))) if target=='boundary_survival' else z.read('selection_probability',int(i)))
                p=np.asarray(weights,float)[valid]
                if valid.sum()<3 or p.sum()<=0:raise ValueError('Insufficient discovery support')
                frames.setdefault(t,[]).append((b,v[valid],p/p.sum(),float(valid.mean())))
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    times=sorted(frames)
    if times!=m['times']:raise ValueError('Incomplete learned time grid')
    if any([b for b,*_ in frames[t]]!=m['splits']['discovery'] for t in times):raise ValueError('Missing discovery batch')
    scale=np.sqrt(time_weights(times)@np.asarray([np.mean([v.var(0) for _,v,_,_ in frames[t]],axis=0) for t in times]))
    if (scale<=1e-12).any() or not np.isfinite(scale).all():raise ValueError('Unestimable motif scale')
    measured=[]
    for t in times:
        modes=[]
        for b,v,p,a in frames[t]:
            sel=dimensionless_moments(v,p,scale);bg=dimensionless_moments(v,np.ones(len(v)),scale)
            modes.append({'center_scaled':sel['center_scaled'],'covariance_dimensionless':sel['covariance_dimensionless'],
                'background_center_scaled':bg['center_scaled'],'background_covariance_dimensionless':bg['covariance_dimensionless'],
                'source_batch':b,'availability':a,'n_available':len(v),'target_weight_ESS':float(1/(p@p))})
        measured.append({'time':t,'modes':modes})
    from .launcher import INPUT_FILES
    ref={'schema_version':'spatial-motif-mixture-1.0','window':m['window'],'times':times,'frames':measured,
        'regions':cat['regions'],'atom_vocabulary':cat['atom_vocabulary'],'channels':list(channels),'channel':'all' if channels==('all',) else 'NOS',
        'spatial_width_A':cat['spatial_width_A'],'spatial_anchor':'endpoint','control_representation':'proposal',
        'representation':'actual native proposal; conditional detached endpoint membership and labels; permutation invariant shell/pair kernels',
        'time_alignment':'score t -> proposal state t+dt; inject only if t+dt within learned window',
        'features':[n.replace('::motif_','::proposal_motif_') for n in names],
        'feature_scale':scale.tolist(),'feature_unit':'mixed: centroid A; kernel densities dimensionless',
        'scale_definition':'fixed whole-window trapezoid mean of equal-batch unweighted within-candidate variance',
        'batches':m['splits']['discovery'],'sources':sources,'source_manifest_sha256':digest(mining/'manifest.json'),
        'target_definition':target,'target_weight_semantics':'Equal discovery batch; normalized same-event selection probability' if target=='instantaneous' else
            'Equal discovery batch; exact descendant counts projected back through actual selection edges whose proposal state is <= learned end. Last outside-state context frame is unweighted and never injected.',
        'required_input_sha256':{name:digest(root/'inputs'/name) for name in INPUT_FILES},
        'limitations':['Associational selection imitation, not verified causal affinity features',
                       'Shell/pair kernels are geometry proxies, not PLIP chemically validated contacts or MMFF energies',
                       'Moments retain alternative batch modes but do not identify complete molecular paths']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    write_json(output.with_suffix('.manifest.json'),{'sha256':digest(output),'bytes':output.stat().st_size,'source_manifest_sha256':ref['source_manifest_sha256'],'target_definition':target})
    return ref


class MotifReward(CoordinateMixtureReward):
    def __init__(self,program,reference):
        self.program,self.reference=program,reference;self.window=reference['window'];self.times=np.asarray(reference['times'])
        if program['window']!=self.window or reference['schema_version']!='spatial-motif-mixture-1.0':raise ValueError('Learning/control contract mismatch')
        self.tau=float(program['mixture_temperature']);self.delta=float(program['robust_delta'])
        self.scale=np.asarray(reference['feature_scale']);self.indices=np.arange(len(self.scale))
        components=program.get('motif_components','joint')
        if components not in ('joint','shell','pair','centroid'):raise ValueError('Unknown motif component')
        if components!='joint':self.indices=np.asarray([i for i,n in enumerate(reference['features']) if components in n])
        if not len(self.indices) or (self.scale<=0).any() or not np.isfinite(self.scale).all() or min(self.tau,self.delta)<=0:raise ValueError('Invalid reward scales')
        if reference['times'][0]!=self.window[0] or reference['times'][-1]!=self.window[-1] or not np.all(np.diff(self.times)>0):raise ValueError('Complete dynamic time grid required')
        if not math.isfinite(program.get('time_ramp_power',0.)) or program.get('time_ramp_power',0.)<0:raise ValueError('Invalid dose schedule')
        for f in reference['frames']:
            for mode in f['modes']:
                for key in ('covariance_dimensionless','background_covariance_dimensionless'):
                    c=np.asarray(mode[key]);np.linalg.cholesky(c)
                    if c.shape!=(len(self.scale),len(self.scale)):raise ValueError('Motif covariance dimension')
    def observables(self,x,atoms,mask,anchor=None):
        if anchor is None:raise ValueError('Detached forecast membership required')
        return torch_motifs(x,atoms,mask,self.reference,anchor,self.reference['spatial_width_A'],self.reference['channels'],self.program['core_radius_A'])
    def __call__(self,x,atoms,mask,time,anchor=None):
        j=int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time)>2e-6:raise ValueError('No measured motif time')
        raw,valid,core=self.observables(x.double(),atoms,mask,anchor.double() if anchor is not None else None)
        ix=self.indices;z=(raw/raw.new_tensor(self.scale))[:,ix];modes=self.reference['frames'][j]['modes']
        def density(background=False):
            ck='background_center_scaled' if background else 'center_scaled';vk='background_covariance_dimensionless' if background else 'covariance_dimensionless'
            center=z.new_tensor(np.asarray([m[ck] for m in modes])[:,ix]);cov=np.asarray([m[vk] for m in modes])[:,ix][:,:,ix]
            precision=z.new_tensor(np.linalg.inv(cov));residual=z[:,None]-center[None]
            q=torch.einsum('bmi,mij,bmj->bm',residual,precision,residual).clamp_min(0)
            if self.program['reward_view']=='motif_contrast':
                logits=-.5*(q+z.new_tensor(np.linalg.slogdet(cov)[1])[None]+len(ix)*math.log(2*math.pi))-math.log(len(modes))
            else:
                cost=self.delta**2*(torch.sqrt(1+(q/len(ix))/self.delta**2)-1)
                logits=-cost/self.tau-math.log(len(modes))
            return torch.logsumexp(logits,1),q,logits
        selected,q,logits=density();r=self.tau*selected;gate=torch.sqrt((q.amin(1)/len(ix))/(1+q.amin(1)/len(ix)))
        if self.program['reward_view']=='motif_contrast':
            background,bq,_=density(True);bound=float(self.program.get('contrast_bound_nats',2.))
            if not math.isfinite(bound) or bound<=0:raise ValueError('Positive contrast bound required')
            u=(selected-background)/bound;r=bound*torch.tanh(u)
            # A near-zero selected/background difference must not acquire a
            # full dose merely because the outer controller normalizes g.
            kl=[]
            for m in modes:
                cs=np.asarray(m['covariance_dimensionless'])[np.ix_(ix,ix)]
                cb=np.asarray(m['background_covariance_dimensionless'])[np.ix_(ix,ix)]
                shift=(np.asarray(m['center_scaled'])-np.asarray(m['background_center_scaled']))[ix]
                kl.append(max(0.,.5*(np.trace(np.linalg.solve(cb,cs))+shift@np.linalg.solve(cb,shift)-len(ix)+np.linalg.slogdet(cb)[1]-np.linalg.slogdet(cs)[1])))
            k=float(np.mean(kl))/len(ix);amplitude=math.sqrt(2*k/(1+2*k))
            gate=(1-torch.tanh(u).square())*amplitude
            # No hard support or composition veto; bounded response controls saturation.
        phase=(time-self.window[0])/(self.window[1]-self.window[0]);power=self.program.get('time_ramp_power',0.)
        # Continuous dose remains positive at every learned node, including startup.
        schedule=(.1+.9*max(0.,min(1.,phase)))**power
        return torch.where(valid,r,raw.sum(1)*0),{'observables':raw,'available':valid,'core_mask':core,
            'nearest_standardized_rms':(q.amin(1)/len(ix)).sqrt(),'dose_gate':torch.where(valid,gate*schedule,0.),
            'mode_responsibilities':logits.softmax(1),'time_dose_factor':raw.new_full((len(raw),),schedule)}

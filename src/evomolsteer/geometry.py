"""Rigid-motion invariant geometric proxies with explicit receptor regions.

No hydrogen bond or aromatic interaction is inferred from distances alone.
"""
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

def pdb_atoms(path):
    rows=[]
    for s in Path(path).read_text().splitlines():
        if not s.startswith('ATOM') or s[16:17] not in (' ','A'): continue
        element=s[76:78].strip() or s[12:16].strip()[0]
        if element in ('H','D'): continue
        rows.append({'chain':s[21:22].strip(),'resid':int(s[22:26]),'icode':s[26:27].strip(),
                     'resname':s[17:20].strip(),'atom':s[12:16].strip(),'element':element,
                     'xyz':[float(s[30:38]),float(s[38:46]),float(s[46:54])]})
    return rows

def sdf_reference_xyz(path):
    lines=Path(path).read_text().splitlines(); n=int(lines[3][:3])
    return np.array([[float(s[:10]),float(s[10:20]),float(s[20:30])] for s in lines[4:4+n]])

def build_catalog(root,config,vocab):
    root=Path(root); regions={}; features={}
    for pocket,pdb,sdf in [('ck2','3PE1_protein_aligned.pdb','3PE1_ligand_aligned.sdf'),
                           ('clk3','6KHF_protein_aligned.pdb','6KHF_ligand_aligned.sdf')]:
        if pocket not in config.get('feature_pockets', ['ck2', 'clk3']):
            continue
        atoms=pdb_atoms(root/'inputs'/pdb); ref=sdf_reference_xyz(root/'inputs'/sdf)
        grouped={}
        for a in atoms:
            key=f'{pocket}:{a["chain"]}:{a["resname"]}{a["resid"]}{a["icode"]}'
            grouped.setdefault(key,[]).append(a)
        for key,aa in sorted(grouped.items()):
            xyz=np.array([a['xyz'] for a in aa])
            if np.linalg.norm(xyz[:,None,:]-ref[None,:,:],axis=-1).min()>config['pocket_radius_A']: continue
            regions[key]={'pocket':pocket,'atoms':aa,'points_A':xyz.tolist()}
            for kind,elements,unit,eligible in [('distance_softmin',[], 'A',True),
                                               ('hetero_distance_softmin',['N','O','S'],'A',True),
                                               ('contact_fraction',[],'fraction',False)]:
                fid=key+'::'+kind
                features[fid]={'id':fid,'region':key,'kind':kind,'elements':elements,'unit':unit,
                               'temperature_A':config['softmin_temperature_A'],
                               'differentiable_supported':eligible,'interpretation':'active ligand slots versus receptor heavy atoms; noisy current/proposal atom labels are not stable chemistry; no validated bond/contact mechanism'}
    features['ligand::radius_gyration']={'id':'ligand::radius_gyration','region':'ligand','kind':'radius_gyration',
        'elements':[],'unit':'A','differentiable_supported':True}
    for name,unit in [('close_pair_fraction','fraction'),('min_pair_distance','A')]:
        fid='ligand::'+name
        features[fid]={'id':fid,'region':'ligand','kind':name,'elements':[],'unit':unit,'differentiable_supported':False}
    return {'schema_version':'1.0','frame':'aligned PDB world coordinates','units':'angstrom',
            'atom_vocabulary':vocab,'regions':regions,'features':features,
            'contact_midpoint_A':config['contact_midpoint_A'],'contact_width_A':config['contact_width_A']}

def measure(xyz,atomics,mask,catalog,chunk_size=256):
    """xyz: [M,N,3] world angstrom; hard atomics used only as conditional masks."""
    xyz=np.asarray(xyz,dtype=np.float64); mask=np.asarray(mask,dtype=bool)
    count=mask.sum(-1); safe=np.maximum(count,1)
    center=np.sum(np.where(mask[...,None],xyz,0),axis=1)/safe[:,None]
    rg=np.sqrt(np.sum(np.where(mask,np.sum((xyz-center[:,None,:])**2,axis=-1),0),axis=-1)/safe)
    rg[count==0]=np.nan
    result={'ligand::radius_gyration':rg}
    pair=np.linalg.norm(xyz[:,:,None,:]-xyz[:,None,:,:],axis=-1)
    pm=mask[:,:,None]&mask[:,None,:]&np.triu(np.ones(pair.shape[1:],bool),k=1)
    denom=pm.sum((1,2)); near=np.sum(pm&(pair<0.8),axis=(1,2))/np.maximum(denom,1)
    mind=np.where(pm,pair,np.inf).min((1,2)); mind[~np.isfinite(mind)]=np.nan; near[denom==0]=np.nan
    result.update({'ligand::close_pair_fraction':near,'ligand::min_pair_distance':mind})
    for region,r in catalog['regions'].items():
        points=np.array(r['points_A'])
        dest={k:[] for k in ['distance_softmin','hetero_distance_softmin','contact_fraction']}
        for start in range(0,len(xyz),chunk_size):
            end=start+chunk_size; xx=xyz[start:end]; mm=mask[start:end]
            dist=np.linalg.norm(xx[:,:,None,:]-points[None,None,:,:],axis=-1)
            for kind in dest:
                spec=catalog['features'][region+'::'+kind]
                em=mm.copy()
                if spec['elements']: em &= np.isin(atomics[start:end],[catalog['atom_vocabulary'][e] for e in spec['elements']])
                if kind=='contact_fraction':
                    val=np.sum(np.where(em,expit((catalog['contact_midpoint_A']-dist.min(-1))/catalog['contact_width_A']),0),axis=-1)/np.maximum(em.sum(-1),1)
                else:
                    tau=spec['temperature_A']; terms=np.where(em[...,None],-dist/tau,-np.inf)
                    n=em.sum(-1)*len(points)
                    val=-tau*(logsumexp(terms,axis=(1,2))-np.log(np.maximum(n,1)))
                val[em.sum(-1)==0]=np.nan; dest[kind].append(val)
        for kind,values in dest.items(): result[region+'::'+kind]=np.concatenate(values)
    return result

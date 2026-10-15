"""Conditional-neighborhood diagnostics; distances are metres, never likelihoods."""
import numpy as np


def reconstruct_history(start,features):
    velocity,acceleration,displacement=np.split(np.asarray(features),3,axis=-1)
    previous=start-velocity/30
    return np.stack((start-displacement,acceleration/900+2*previous-start,previous,start))


def trajectory_distance(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b),axis=-1).mean(axis=(-1,-2))


def neighborhood_metrics(futures,prediction,truth):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    f=np.asarray(futures,np.float64);pred=np.asarray(prediction,np.float64);gt=np.asarray(truth,np.float64)
    if f.ndim!=4 or f.shape[1:]!=(24,11,3) or len(f)<8:raise ValueError('>=8 complete hand futures required')
    mean=f.mean(0);pd=trajectory_distance(f,pred);gd=trajectory_distance(f,gt)
    result=dict(neighbors=len(f),spread_m=float(trajectory_distance(f,mean).mean()),
        pred_mean_m=float(trajectory_distance(pred,mean)),pred_real_m=float(pd.min()),
        pred_nn3_m=float(np.sort(pd)[:3].mean()),gt_nn3_m=float(np.sort(gd)[:3].mean()),
        mean_nn3_m=float(np.sort(trajectory_distance(f,mean))[:3].mean()),
        gt_mean_m=float(trajectory_distance(gt,mean)),multimodal=False)
    # TRAIN-only descriptor: wrist and five tips at first8/final24; no query GT clustering.
    descriptor=f[:,[7,23]][:,:,[0,6,7,8,9,10]].reshape(len(f),-1)/.05
    if np.max(np.std(descriptor,axis=0))<1e-5:return result
    x=PCA(n_components=min(3,len(f)-2,descriptor.shape[1]),svd_solver='full').fit_transform(descriptor)
    one=GaussianMixture(1,covariance_type='diag',reg_covar=1e-4,random_state=214,n_init=2).fit(x)
    two=GaussianMixture(2,covariance_type='diag',reg_covar=1e-4,random_state=214,n_init=3).fit(x)
    labels=two.predict(x);counts=np.bincount(labels,minlength=2)
    if counts.min()<2:return result
    prototypes=np.stack([f[labels==k].mean(0) for k in (0,1)])
    medoids=np.stack([f[labels==k][np.argmin(trajectory_distance(f[labels==k],prototypes[k]))] for k in (0,1)])
    gap=float(one.bic(x)-two.bic(x));sil=float(silhouette_score(f.reshape(len(f),-1),labels))
    separation=float(trajectory_distance(prototypes[0],prototypes[1]))
    dm=float(trajectory_distance(prototypes,pred).min());dmed=float(trajectory_distance(medoids,pred).min())
    multi=bool(gap>=10 and sil>=.5 and counts.min()>=max(3,.2*len(f)) and separation>=.02)
    result.update(bic_gain=gap,silhouette=sil,mode_counts=counts.tolist(),mode_separation_m=separation,
        pred_mode_m=dm,pred_medoid_m=dmed,multimodal=multi,
        between_modes=bool(dm-result['pred_mean_m']>=.005),
        off_empirical_support=bool(result['pred_nn3_m']>1.5*max(result['gt_nn3_m'],.001)),
        averaging_score=dm/(dm+result['pred_mean_m']+1e-12))
    return result


def motion_metrics(pred_delta,gt_delta):
    pred=np.asarray(pred_delta,np.float64);gt=np.asarray(gt_delta,np.float64)
    error=np.linalg.norm(pred-gt,axis=-1)
    result=dict(ade_m=float(error.mean()),ade8_m=float(error[:8].mean()),fde_m=float(error[-1].mean()))
    for name,part in [('first8',slice(0,8)),('all24',slice(None)),('h24',slice(23,24))]:
        a=np.linalg.norm(pred[part],axis=-1).mean();b=np.linalg.norm(gt[part],axis=-1).mean()
        result[name+'_gt_motion_m']=float(b);result[name+'_motion_ratio']=float(a/b) if b>=.005 else None
    pv=np.diff(np.concatenate((np.zeros_like(pred[:1]),pred)),axis=0)*30
    gv=np.diff(np.concatenate((np.zeros_like(gt[:1]),gt)),axis=0)*30
    for name,a,b in [('speed',pv,gv),('acceleration',np.diff(pv,axis=0)*30,np.diff(gv,axis=0)*30)]:
        a=np.linalg.norm(a,axis=-1).mean();b=np.linalg.norm(b,axis=-1).mean()
        result['gt_'+name]=float(b);result[name+'_ratio']=float(a/b) if b>1e-5 else None
    return result


def choose_neighbors(query,bank,metadata_q,metadata_b,zdist,gate,k=32):
    """Match input conditions only, then distinct captures; future arrays never enter."""
    candidates=[]
    for j,m in enumerate(metadata_b):
        if m['source_id']!=metadata_q['source_id'] or m['side']!=metadata_q['side'] or m['hand_pattern']!=metadata_q['hand_pattern']:continue
        if m['geometry_key']!=metadata_q['geometry_key'] or m['scene_signature']!=metadata_q['scene_signature']:continue
        if m['capture']==metadata_q['capture']:continue
        mask=((int(metadata_q['hand_pattern'])>>np.arange(22))&1).astype(bool).reshape(2,11)
        h=float(np.sqrt(np.mean((bank['history'][j][:,mask]-query['history'][:,mask])**2)))
        g=float(np.sqrt(np.mean((bank['goal'][j]-query['goal'])**2)))
        n=metadata_q.get('n_objects',1)
        scene=float(np.sqrt(np.mean((bank['scene_history'][j,:n]-query['scene_history'][:n])**2)))
        # RMS in Euclidean metres, across XYZ as a vector (not coordinate RMS).
        h*=3**.5;g*=3**.5;scene*=3**.5
        if max(h,g,scene)<=gate:candidates.append((j,h,g,scene))
    candidates.sort(key=(lambda x:float(zdist[x[0]])) if zdist is not None else (lambda x:x[1]**2+x[2]**2+x[3]**2))
    picked=[];seen=set()
    for j,h,g,s in candidates:
        capture=metadata_b[j]['capture']
        if capture in seen:continue
        picked.append((j,h,g,s));seen.add(capture)
        if len(picked)==k:break
    return picked

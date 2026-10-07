"""Memory-only adapters for original FoundationPose at native RGB resolution."""
import torch


def transform_pose_batch_in_chunks(transform, dataset, batch, chunk_size=32, **kwargs):
    """Apply the original per-pose transform in chunks, retaining every pose.

    FoundationPose backprojects each candidate crop to the original image
    resolution. Its888-candidate1080p batch exceeds a24GB GPU. This changes
    only simultaneous candidate count; selection, normalization and outputs
    are the original operations. No candidate is removed or reordered.
    """
    count = len(batch.rgbAs)
    if count <= chunk_size:
        return transform(dataset,batch,**kwargs)
    chunks=[]
    for start in range(0,count,chunk_size):
        selected=batch.select_by_indices(torch.arange(start,min(count,start+chunk_size)))
        chunks.append(transform(dataset,selected,**kwargs))
    result=type(batch)()
    for name,value in chunks[0].__dict__.items():
        values=[part.__dict__[name] for part in chunks]
        if value is None:
            if any(x is not None for x in values):
                raise ValueError('inconsistent optional pose fields')
            result.__dict__[name]=None
        else:
            result.__dict__[name]=torch.cat(values,dim=0)
    return result


def install_native_resolution_batching(chunk_size=32):
    from learning.datasets.h5_dataset import TripletH5Dataset
    original=TripletH5Dataset.transform_batch
    def transform(dataset,batch,H_ori,W_ori,bound=1):
        return transform_pose_batch_in_chunks(original,dataset,batch,chunk_size,
                                             H_ori=H_ori,W_ori=W_ori,bound=bound)
    TripletH5Dataset.transform_batch=transform

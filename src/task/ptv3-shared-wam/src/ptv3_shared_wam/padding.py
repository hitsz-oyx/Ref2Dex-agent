"""Task-local equivalent of PTv3's serialized padding without per-row CUDA scalars."""
import torch


@torch.no_grad()
def vectorized_padding(self, point, prefix=''):
    keys = [prefix+'pad',prefix+'unpad',prefix+'cu_seqlens_key']
    if not all(key in point for key in keys):
        offset = point.offset
        starts = torch.cat((offset.new_zeros(1),offset[:-1]))
        counts = offset-starts
        patch = self.patch_size
        padded_counts = torch.where(counts>patch,((counts+patch-1)//patch)*patch,counts)
        ends = padded_counts.cumsum(0)
        padded_starts = ends-padded_counts
        total_padded = int(ends[-1])
        padded = torch.arange(total_padded,device=offset.device)
        group = torch.bucketize(padded,ends,right=True)
        local = padded-padded_starts[group]
        # Reuse the preceding full patch for a partial final patch, exactly as
        # upstream; never duplicate a different sample's points.
        local = torch.where(local>=counts[group],local-patch,local)
        pad = starts[group]+local
        original = torch.arange(point.feat.shape[0],device=offset.device)
        original_group = torch.bucketize(original,offset,right=True)
        unpad = original+(padded_starts-starts)[original_group]
        chunks = (padded_counts+patch-1)//patch
        chunk_ends = chunks.cumsum(0)
        chunk_starts = chunk_ends-chunks
        total_chunks = int(chunk_ends[-1])
        chunk_ids = torch.arange(total_chunks,device=offset.device)
        chunk_group = torch.bucketize(chunk_ids,chunk_ends,right=True)
        cu = padded_starts[chunk_group]+(chunk_ids-chunk_starts[chunk_group])*patch
        point[keys[0]],point[keys[1]],point[keys[2]] = pad,unpad,torch.cat((cu,ends[-1:])).int()
    return tuple(point[key] for key in keys)

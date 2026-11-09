"""Lazy CPU-array adapter in the MODEL interpreter; never import frozen inputs."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import io
from pathlib import Path
from . import training as t


class TrainingData:
    def __init__(self):
        self.rows=t.load(t.INPUT/"requests.json")
        self.items={r['id']:r for r in t.load(t.INPUT/"native_inputs.json")}
        self.sources=t.load(t.INPUT/"source_refs.json")
        self.materials=t.load(t.c.OUT/"selected_materials.json")
        self.compat=t.load(t.INPUT/"compatibility.json")["inputs"]

    def arrays(self,path,keys):
        import numpy as np
        path=t.c.no_redirect(path)
        t.c.require(str(path) in self.sources and path.suffix=='.npz' and path.stat().st_size<=16*2**20,"Unbound training NPZ")
        raw=path.read_bytes()
        t.c.require(hashlib.sha256(raw).hexdigest()==self.sources[str(path)],"NPZ same-byte source hash differs")
        with np.load(io.BytesIO(raw),allow_pickle=False) as p:
            t.c.require(set(p.files)==set(keys),"Training NPZ keys differ")
            return {n:p[n] for n in keys}

    @lru_cache(maxsize=64)
    def native(self,sample_id):
        import numpy as np
        row=self.items[sample_id]; p=self.arrays(t.NATIVE/row['cache'],['rgb','mask'])
        rgb,mask=p['rgb'],p['mask']; h,w=row['height'],row['width']
        t.c.require(rgb.dtype==np.uint8 and rgb.shape==(h,w,3) and mask.dtype==np.uint8 and mask.shape==(h,w),"Native dtype/shape differs")
        t.c.require(np.all((mask==0)|(mask==1)) and bool(mask.any())==(row['kind']=='real'),"Native binary role differs")
        rgb.flags.writeable=False; mask.flags.writeable=False
        return rgb,mask

    @staticmethod
    def normalize(image):
        import numpy as np
        t.c.require(image.dtype==np.float32 and image.shape==(256,256,3) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,"Expected unquantized float32[0,1]")
        return np.ascontiguousarray(((image-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)).transpose(2,0,1))

    @staticmethod
    def identity(pack):
        import numpy as np
        result={}; h=hashlib.sha256()
        for key in ['image','target','valid']:
            array=np.ascontiguousarray(pack[key]); label=f"{key}:{array.dtype}:{array.shape}".encode()
            value=hashlib.sha256(array.tobytes()).hexdigest(); result[key]=value
            h.update(label); h.update(bytes.fromhex(value))
        result['pack']=h.hexdigest()
        return result

    def pack(self,ref,flip_h=False,flip_v=False):
        import numpy as np
        if ref['kind']=='synthetic':
            row=self.materials[ref['ordinal']]
            t.c.require(row['ordinal']==ref['ordinal'] and row['type']==ref['type'],"Synthetic ordinal/type differs")
            p=self.arrays(Path(row['file']),['image','target','valid'])
            image,target,valid=p['image'],p['target'],p['valid']
            t.c.require(target.dtype==valid.dtype==np.uint8 and target.shape==valid.shape==(256,256) and np.all((target==0)|(target==1)) and np.all((valid==0)|(valid==1)),"Synthetic binary mask/domain differs")
            valid=valid.astype(bool)
            t.c.require(target.any() and not target[~valid].any(),"Synthetic target outside effective domain")
            # Already reflected and flipped in the OLD accepted cache. No /255, re-padding or flip.
            return {'image':self.normalize(image),'target':np.ascontiguousarray(target),'valid':np.ascontiguousarray(valid)}
        row=self.items[ref['id']]; rgb,mask=self.native(ref['id']); y,w=ref['y'],row['width']
        t.c.require(ref['kind']==row['kind'] and y in [r['y'] for r in row['windows']],"Native role/window differs")
        rgb=np.pad(rgb[y:y+256],((0,0),(0,256-w),(0,0)),mode='reflect')
        target=np.pad(mask[y:y+256],((0,0),(0,256-w)),mode='constant')
        valid=np.zeros((256,256),dtype=bool); valid[:,:w]=True
        for axis,active in [(1,flip_h),(0,flip_v)]:
            if active: rgb,target,valid=(np.flip(x,axis=axis).copy() for x in [rgb,target,valid])
        return {'image':self.normalize(rgb.astype(np.float32)/np.float32(255)),'target':np.ascontiguousarray(target),'valid':np.ascontiguousarray(valid)}

    def request(self,group,update):
        t.c.require(group in {'A','B'} and type(update) is int and 1<=update<=10000,"Invalid training input cursor")
        row=self.rows[update-1]; ref=row.get('common_input',row.get(group+'_input'))
        pack=self.pack(ref,row['flip_h'],row['flip_v'])
        t.c.require(self.identity(pack)==row['input_identity_'+group],"Runtime normalized arrays differ from frozen CPU identities; stop, do not relax hash")
        return pack,row,ref

    def compatibility(self,index):
        row=self.compat[index]; ref={k:row[k] for k in (['kind','ordinal','type'] if row['kind']=='synthetic' else ['kind','id','y'])}
        pack=self.pack(ref)
        t.c.require(self.identity(pack)==row['input_identity'],"Compatibility input bytes differ from frozen CPU metadata")
        return pack,row

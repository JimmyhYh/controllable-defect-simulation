"""Future scoring format and native-window stitch; no calibration/test entry yet."""
import numpy as np
from . import common as c


SCORE_FORMAT = {'dtype':'float32','range':[0,1],'channels':'anomaly softmax',
                'padding_excluded':True,'window':256,'stride':128,'overlap':'equal mean',
                'score':'max native stitched map','q':.95,'quantile_method':'linear','positive':'strict >'}


def stitch_native(height,width,window_probabilities):
    """CPU stitch only; consumes exactly all y-ordered, unflipped maps."""
    c.require(2 <= width < 256 and height >=256,'Unexpected native dimensions')
    starts=c.d.window_starts(height)
    total=np.zeros((height,width),np.float32); count=np.zeros((height,width),np.uint16)
    seen=[]
    for y,probability in window_probabilities:
        c.require(y in starts and y not in seen,'Duplicate/unexpected inference window')
        p=np.asarray(probability)
        c.require(p.shape==(256,256) and p.dtype==np.float32 and np.isfinite(p).all()
                  and p.min()>=0 and p.max()<=1,'Invalid raw probability map')
        total[y:y+256]+=p[:,:width]; count[y:y+256]+=1; seen.append(y)
    c.require(seen==starts and np.all(count>0),'Missing/out-of-order full-image windows')
    return (total/count).astype(np.float32)

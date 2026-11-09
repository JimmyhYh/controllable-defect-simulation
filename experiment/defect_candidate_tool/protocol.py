"""Metadata and scalar contract; no image or model imports."""
import math

CLASSES = {1: '块状亮度不均', 2: '线状亮度扰动'}
DEFAULTS = dict(amplitude=0.04, edge_width_px=4.0, variation=0.3,
                batch_size=6, base_seed=2026110901)

def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric scalar required')
    return float(value)

def point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Expected [x,y]')
    return tuple(number(v) for v in value)

def validate_request(request: dict, height: int, width: int) -> dict:
    if any(type(v) is not int or v < 8 for v in (height, width)):
        raise ValueError('Native height/width must be integers >=8')
    common = {'class_id','shape','polarity','amplitude','edge_width_px','variation','batch_size','base_seed'}
    shapes = {'ellipse': {'center_xy','semiaxes_px','angle_deg'},
              'irregular_block': {'center_xy','semiaxes_px','angle_deg'},
              'line': {'center_xy','length_px','width_px','angle_deg'},
              'polyline': {'points_xy','width_px'}}
    shape = request.get('shape')
    if shape not in shapes or set(request) != common | shapes[shape]:
        raise ValueError('Missing/unknown fields or unsupported shape')
    cid = request['class_id']
    if type(cid) is not int or cid not in CLASSES or (cid == 1) != (shape in ('ellipse','irregular_block')):
        raise ValueError('Category/shape mismatch')
    if request['polarity'] not in ('bright','dark'):
        raise ValueError('Invalid polarity')
    if not 0 < number(request['amplitude']) <= .25 or not 0 <= number(request['variation']) <= 1:
        raise ValueError('Amplitude/variation outside contract')
    if number(request['edge_width_px']) <= 0:
        raise ValueError('Positive edge width required')
    if type(request['batch_size']) is not int or not 1 <= request['batch_size'] <= 64:
        raise ValueError('Batch size must be 1..64; sequential generation')
    if type(request['base_seed']) is not int or not 0 <= request['base_seed'] < 2**64:
        raise ValueError('Seed must be uint64')
    if shape in ('ellipse','irregular_block'):
        cx,cy=point(request['center_xy']); a,b=point(request['semiaxes_px'])
        if min(a,b) <= 1: raise ValueError('Semiaxes must exceed one pixel')
        theta=math.radians(number(request['angle_deg'])); c,s=math.cos(theta),math.sin(theta)
        ex,ey=math.hypot(a*c,b*s),math.hypot(a*s,b*c)
        lo,hi=(cx-ex,cy-ey),(cx+ex,cy+ey)
    else:
        thickness=number(request['width_px'])
        if thickness <= 1: raise ValueError('Width must exceed one pixel')
        if shape=='line':
            cx,cy=point(request['center_xy']); length=number(request['length_px'])
            if length < 3: raise ValueError('Line length must be >=3')
            theta=math.radians(number(request['angle_deg'])); dx,dy=math.cos(theta)*length/2,math.sin(theta)*length/2
            points=[(cx-dx,cy-dy),(cx+dx,cy+dy)]
        else:
            if not isinstance(request['points_xy'],list) or not 2 <= len(request['points_xy']) <= 16:
                raise ValueError('Polyline needs 2..16 points')
            points=[point(v) for v in request['points_xy']]
        if any(math.dist(a,b)<3 for a,b in zip(points,points[1:])):
            raise ValueError('Path segments must be >=3 pixels')
        lo=(min(p[0] for p in points)-thickness/2,min(p[1] for p in points)-thickness/2)
        hi=(max(p[0] for p in points)+thickness/2,max(p[1] for p in points)+thickness/2)
    if lo[0]<0 or lo[1]<0 or hi[0]>width-1 or hi[1]>height-1:
        raise ValueError('Geometry outside native frame; no clipping/relocation')
    return dict(request)

def contract():
    return {'schema':'defect-candidate-input-v1','classes':CLASSES,'defaults':DEFAULTS,
            'coordinate':'EXIF-normalized native pixel centers, x/y, top-left origin; no resize',
            'accepted_formats':['PNG','JPEG'],'accepted_modes':['RGB','L'],
            'reject':['alpha including opaque RGBA','palette/tRNS','16-bit','unknown mode'],
            'normalization':'EXIF transpose then L replicated into RGB, no contrast stretch',
            'class_background':0,'confirmed_class':None,'instances_per_request':1,
            'masks':{'M_changed':'D>0','M_eff':'D>float32(1/255)','M_png':'uint8 max channel code difference>0'},
            'residual':'max absolute RGB channel difference; encoded RGB, not luminance/nit',
            'resources':{'max_native_pixels':4194304,'estimated_peak_bytes_per_pixel':192,
                         'fixed_overhead_bytes':268435456,'reserve_bytes':536870912,
                         'disk_min_free_bytes':2147483648,'batch_sequential':True,
                         'estimate_not_measured_peak':True},
            'phase2_input_decode_and_transparency_check_required':True}

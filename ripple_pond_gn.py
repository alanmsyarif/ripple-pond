# ============================================================
#  RIPPLE FORGE GN v0.3 — Geometry Nodes interactive pond
#  + Weta-style whitewater particles (bubbles / foam / spray)
#
#  Architecture:
#   * Wave solver:  GN Simulation Zone on the water grid.
#     Blur Attribute is used as a discrete Laplacian, giving the
#     2D wave equation  h_new = 2h - h_prev + c^2*lap  entirely
#     in nodes. Interaction via Geometry Proximity to a collection
#     (objects displace the surface where they intersect it).
#   * Whitewater:  GN Simulation Zone point system following
#     Wretborn, Flynn, Stomakhin 2022 "Guided Bubbles and Wet
#     Foam for Realistic Whitewater Simulation" (Weta):
#       - three particle classes: bubble(0) / foam(1) / spray(2)
#       - emission from an aeration proxy (curvature + surface
#         velocity fluctuation), bubble radii drawn from the
#         paper's inverse-cubic distribution R^-1(X)
#       - bubbles: buoyancy ~ r^2 vs drag toward the guide
#         velocity field (surface slope + dh/dt)
#       - wet foam: manifold-constrained to the surface (z-pin),
#         surface drag toward liquid velocity, pairwise
#         cohesion/pressure using effective distance |x|-(rp+rq)
#         (paper eq. 43), wetness-extended lifespan
#       - transitions per the paper's Algorithm 2 with tangential
#         momentum preservation (zeta = 0.7)
#   * Everything is adjustable live in the viewport via modifier
#     sliders. Sim runs on timeline playback (spacebar).
#
#  Install: Edit > Preferences > Add-ons > Install > this file
#  Requires Blender 4.2+ (Simulation Zones, Index of Nearest)
# ============================================================

bl_info = {
    "name": "Ripple Forge GN",
    "author": "Amsy",
    "version": (0, 3, 3),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Ripple Forge",
    "description": "Geometry-Nodes interactive pond with Weta-style "
                   "whitewater particles (guided bubbles + wet foam)",
    "category": "Object",
}

import bpy
import math
from mathutils import Vector
from bpy_extras import view3d_utils

WATER_OBJ  = "RF_Water"
FLOOR_OBJ  = "RF_Floor"
WW_OBJ     = "RF_Whitewater"
BRUSH_OBJ  = "RF_Brush"
COL_INTER  = "RF_Interactors"
COL_OBST   = "RF_Obstacles"
WATER_TREE = "RF_WaterSim"
WW_TREE    = "RF_WhitewaterSim"
FLOOR_TREE = "RF_FloorWetness"
WATER_MAT  = "RF_Water_Mat"
FLOOR_MAT  = "RF_Floor_Mat"
WW_MAT     = "RF_Whitewater_Mat"


# ------------------------------------------------------------
# Node tree builder helper
# ------------------------------------------------------------

class NT:
    """Compact helper for programmatic Geometry Nodes construction.
    Socket arguments may be NodeSocket objects (linked) or python
    values (set as default_value)."""

    def __init__(self, tree):
        self.t = tree

    def n(self, bl_idname, x=0, y=0, **props):
        node = self.t.nodes.new(bl_idname)
        node.location = (x, y)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def l(self, out_sock, in_sock):
        self.t.links.new(out_sock, in_sock)

    def w(self, in_sock, val):
        """Wire a socket or set a constant (with int/bool coercion)."""
        if isinstance(val, bpy.types.NodeSocket):
            self.l(val, in_sock)
        elif val is not None:
            if in_sock.type == 'INT' and isinstance(val, float):
                in_sock.default_value = int(val)
            elif in_sock.type == 'BOOLEAN' and isinstance(val, (int, float)):
                in_sock.default_value = bool(val)
            else:
                in_sock.default_value = val

    @staticmethod
    def find(sockets, name, stype=None):
        """Resolve a socket by name and (optionally) type — robust against
        socket-order changes between Blender releases."""
        for s in sockets:
            if s.name == name and (stype is None or s.type == stype):
                return s
        for s in sockets:  # fallback: name only
            if s.name == name:
                return s
        return None

    _UNARY = {'LENGTH', 'NORMALIZE', 'ABSOLUTE', 'SQRT', 'SINE', 'COSINE',
              'FLOOR', 'CEIL', 'FRACT', 'SIGN', 'ROUND', 'INVERSE_SQRT'}

    # ---- float math ----
    def m(self, op, a, b=None, x=0, y=0):
        node = self.n('ShaderNodeMath', x, y, operation=op)
        self.w(node.inputs[0], a)
        if b is not None and op not in self._UNARY:
            self.w(node.inputs[1], b)
        return node.outputs[0]

    def madd(self, a, b, c, x=0, y=0):  # a*b + c
        node = self.n('ShaderNodeMath', x, y, operation='MULTIPLY_ADD')
        self.w(node.inputs[0], a)
        self.w(node.inputs[1], b)
        self.w(node.inputs[2], c)
        return node.outputs[0]

    # ---- vector math ----
    _UNARY_V = {'LENGTH', 'NORMALIZE', 'ABSOLUTE', 'FRACTION',
                'FLOOR', 'CEIL', 'SINE', 'COSINE', 'TANGENT'}

    def vm(self, op, a, b=None, x=0, y=0):
        # unary ops take no second operand — if a numeric slipped into b,
        # it was meant as the x coordinate
        if op in self._UNARY_V and isinstance(b, (int, float)):
            b, x, y = None, b, x
        node = self.n('ShaderNodeVectorMath', x, y, operation=op)
        self.w(node.inputs[0], a)
        if b is not None and op not in self._UNARY_V:
            self.w(node.inputs[1], b)
        if op == 'LENGTH':
            return self.find(node.outputs, 'Value', 'VALUE')
        return self.find(node.outputs, 'Vector', 'VECTOR')

    def vscale(self, v, s, x=0, y=0):
        node = self.n('ShaderNodeVectorMath', x, y, operation='SCALE')
        self.w(node.inputs[0], v)
        self.w(self.find(node.inputs, 'Scale', 'VALUE'), s)
        return self.find(node.outputs, 'Vector', 'VECTOR')

    # ---- mix ----
    def fmix(self, fac, a, b, x=0, y=0):
        node = self.n('ShaderNodeMix', x, y, data_type='FLOAT')
        self.w(self.find(node.inputs, 'Factor', 'VALUE'), fac)
        self.w(self.find(node.inputs, 'A', 'VALUE'), a)
        self.w(self.find(node.inputs, 'B', 'VALUE'), b)
        return self.find(node.outputs, 'Result', 'VALUE')

    def vmix(self, fac, a, b, x=0, y=0):
        node = self.n('ShaderNodeMix', x, y, data_type='VECTOR')
        node.factor_mode = 'UNIFORM'
        fsock = self.find(node.inputs, 'Factor', 'VALUE') \
            or self.find(node.inputs, 'Factor', 'VECTOR')
        self.w(fsock, fac)
        self.w(self.find(node.inputs, 'A', 'VECTOR'), a)
        self.w(self.find(node.inputs, 'B', 'VECTOR'), b)
        return self.find(node.outputs, 'Result', 'VECTOR')

    # ---- logic ----
    def cmp(self, op, a, b, x=0, y=0, eps=None):
        node = self.n('FunctionNodeCompare', x, y,
                      data_type='FLOAT', operation=op)
        self.w(self.find(node.inputs, 'A', 'VALUE'), a)
        self.w(self.find(node.inputs, 'B', 'VALUE'), b)
        es = self.find(node.inputs, 'Epsilon', 'VALUE')
        if eps is not None and es is not None:
            es.default_value = eps
        return node.outputs['Result']

    def band(self, a, b, x=0, y=0):
        node = self.n('FunctionNodeBooleanMath', x, y, operation='AND')
        self.w(node.inputs[0], a)
        self.w(node.inputs[1], b)
        return node.outputs[0]

    def bor(self, a, b, x=0, y=0):
        node = self.n('FunctionNodeBooleanMath', x, y, operation='OR')
        self.w(node.inputs[0], a)
        self.w(node.inputs[1], b)
        return node.outputs[0]

    def bnot(self, a, x=0, y=0):
        node = self.n('FunctionNodeBooleanMath', x, y, operation='NOT')
        self.w(node.inputs[0], a)
        return node.outputs[0]

    def sw(self, cond, false_v, true_v, x=0, y=0, input_type='FLOAT'):
        node = self.n('GeometryNodeSwitch', x, y, input_type=input_type)
        self.w(node.inputs['Switch'], cond)
        self.w(node.inputs['False'], false_v)
        self.w(node.inputs['True'], true_v)
        return node.outputs[0]

    # ---- attributes ----
    def na(self, name, dtype='FLOAT', x=0, y=0):
        node = self.n('GeometryNodeInputNamedAttribute', x, y,
                      data_type=dtype)
        node.inputs['Name'].default_value = name
        return node.outputs['Attribute']

    _TYPEMAP = {'FLOAT': 'VALUE', 'FLOAT_VECTOR': 'VECTOR',
                'BOOLEAN': 'BOOLEAN', 'INT': 'INT'}

    def store(self, geo, name, value, dtype='FLOAT', x=0, y=0):
        node = self.n('GeometryNodeStoreNamedAttribute', x, y,
                      data_type=dtype, domain='POINT')
        self.l(geo, node.inputs['Geometry'])
        node.inputs['Name'].default_value = name
        self.w(self.find(node.inputs, 'Value', self._TYPEMAP.get(dtype)),
               value)
        return node.outputs['Geometry']

    # attribute data-type -> socket-type vocabulary (capture_items API)
    _CAPMAP = {'FLOAT': 'FLOAT', 'FLOAT_VECTOR': 'VECTOR',
               'BOOLEAN': 'BOOLEAN', 'INT': 'INT'}

    def capture(self, geo, value, dtype='FLOAT', x=0, y=0):
        node = self.n('GeometryNodeCaptureAttribute', x, y, domain='POINT')
        if hasattr(node, 'capture_items'):        # Blender 4.2+ multi-item API
            try:
                node.capture_items.new(self._CAPMAP.get(dtype, dtype), "value")
            except TypeError:                     # older enum vocabulary
                node.capture_items.new(dtype, "value")
        else:                                     # legacy single-item API
            node.data_type = dtype
        self.l(geo, node.inputs['Geometry'])
        v_in = next(s for s in node.inputs if s.type != 'GEOMETRY' and s.enabled)
        v_out = next(s for s in node.outputs if s.type != 'GEOMETRY' and s.enabled)
        self.w(v_in, value)
        return node.outputs['Geometry'], v_out

    # ---- misc ----
    def rand(self, vmin, vmax, seed, x=0, y=0):
        node = self.n('FunctionNodeRandomValue', x, y, data_type='FLOAT')
        self.w(self.find(node.inputs, 'Min', 'VALUE'), vmin)
        self.w(self.find(node.inputs, 'Max', 'VALUE'), vmax)
        s = self.find(node.inputs, 'Seed', 'INT')
        if s is not None:
            s.default_value = int(seed)
        return self.find(node.outputs, 'Value', 'VALUE')

    def sep(self, v, x=0, y=0):
        node = self.n('ShaderNodeSeparateXYZ', x, y)
        self.w(node.inputs[0], v)
        return node.outputs  # X, Y, Z

    def comb(self, xx, yy, zz, x=0, y=0):
        node = self.n('ShaderNodeCombineXYZ', x, y)
        self.w(node.inputs[0], xx)
        self.w(node.inputs[1], yy)
        self.w(node.inputs[2], zz)
        return node.outputs[0]

    def pos(self, x=0, y=0):
        return self.n('GeometryNodeInputPosition', x, y).outputs[0]

    # sample a named attr from a mesh at given positions
    def sns(self, mesh_geo, attr_name, dtype, sample_pos, x=0, y=0):
        val = self.na(attr_name, dtype, x - 180, y)
        node = self.n('GeometryNodeSampleNearestSurface', x, y,
                      data_type=dtype)
        self.l(mesh_geo, node.inputs['Mesh'])
        self.w(self.find(node.inputs, 'Value', self._TYPEMAP.get(dtype)),
               val)
        self.w(node.inputs['Sample Position'], sample_pos)
        return self.find(node.outputs, 'Value', self._TYPEMAP.get(dtype))


def _new_tree(name):
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.is_modifier = True
    return tree


def _iface(tree, name, in_out, stype, default=None, minv=None, maxv=None):
    s = tree.interface.new_socket(name=name, in_out=in_out,
                                  socket_type=stype)
    if default is not None:
        s.default_value = default
    if minv is not None:
        s.min_value = minv
    if maxv is not None:
        s.max_value = maxv
    return s

# ------------------------------------------------------------
# Water surface simulation tree (wave eq in a Simulation Zone)
# ------------------------------------------------------------

def build_water_tree():
    tree = _new_tree(WATER_TREE)
    b = NT(tree)

    _iface(tree, "Geometry",  'INPUT',  'NodeSocketGeometry')
    _iface(tree, "Geometry",  'OUTPUT', 'NodeSocketGeometry')
    _iface(tree, "Wave Speed",       'INPUT', 'NodeSocketFloat', 0.25, 0.0, 0.5)
    _iface(tree, "Damping",          'INPUT', 'NodeSocketFloat', 0.995, 0.9, 1.0)
    _iface(tree, "Amplitude",        'INPUT', 'NodeSocketFloat', 1.0, 0.0, 10.0)
    _iface(tree, "Interactors",      'INPUT', 'NodeSocketCollection')
    _iface(tree, "Interact Radius",  'INPUT', 'NodeSocketFloat', 0.25, 0.01, 5.0)
    _iface(tree, "Interact Depth",   'INPUT', 'NodeSocketFloat', 0.05, 0.0, 1.0)
    _iface(tree, "Obstacles",        'INPUT', 'NodeSocketCollection')
    _iface(tree, "Obstacle Radius",  'INPUT', 'NodeSocketFloat', 0.3, 0.01, 5.0)
    _iface(tree, "Foam Gain",        'INPUT', 'NodeSocketFloat', 60.0, 0.0, 500.0)
    _iface(tree, "Foam Decay",       'INPUT', 'NodeSocketFloat', 0.96, 0.5, 1.0)
    _iface(tree, "Wet Decay",        'INPUT', 'NodeSocketFloat', 0.995, 0.9, 1.0)
    _iface(tree, "Wet Sensitivity",  'INPUT', 'NodeSocketFloat', 0.01, 0.001, 0.5)

    gin  = b.n('NodeGroupInput', -1600, 0)
    gin2 = b.n('NodeGroupInput', -400, -700)
    gout = b.n('NodeGroupOutput', 1800, 0)

    sim_in  = b.n('GeometryNodeSimulationInput', -1300, 0)
    sim_out = b.n('GeometryNodeSimulationOutput', 1000, 0)
    sim_in.pair_with_output(sim_out)
    b.l(gin.outputs['Geometry'], sim_in.inputs['Geometry'])

    geo = sim_in.outputs['Geometry']

    # --- previous state ---
    h    = b.na("h",    'FLOAT', -1100, 200)
    hp   = b.na("hp",   'FLOAT', -1100, 60)
    foam = b.na("foam", 'FLOAT', -1100, -80)
    wet  = b.na("wet",  'FLOAT', -1100, -220)

    # --- laplacian via Blur Attribute ---
    blur_n = b.n('GeometryNodeBlurAttribute', -900, 240, data_type='FLOAT')
    b.w(blur_n.inputs['Value'], h)
    blur_n.inputs['Iterations'].default_value = 1
    blur_n.inputs['Weight'].default_value = 1.0
    lap = b.m('SUBTRACT', blur_n.outputs[0], h, -700, 240)

    # --- wave equation:  hn = 2h - hp + speed*lap ---
    two_h  = b.m('MULTIPLY', h, 2.0, -700, 120)
    base   = b.m('SUBTRACT', two_h, hp, -540, 140)
    hn     = b.madd(lap, gin.outputs['Wave Speed'], base, -380, 180)
    hn     = b.m('MULTIPLY', hn, gin.outputs['Damping'], -220, 180)

    # --- interaction: objects intersecting the surface push it down ---
    ci = b.n('GeometryNodeCollectionInfo', -1100, -420,
             transform_space='RELATIVE')
    b.w(ci.inputs['Collection'], gin.outputs['Interactors'])
    ri = b.n('GeometryNodeRealizeInstances', -920, -420)
    b.l(ci.outputs[0], ri.inputs[0])
    prox = b.n('GeometryNodeProximity', -740, -420,
               target_element='FACES')
    b.l(ri.outputs[0], prox.inputs[0])
    contact = b.n('ShaderNodeMapRange', -560, -420,
                  interpolation_type='SMOOTHSTEP')
    b.w(contact.inputs['Value'], prox.outputs['Distance'])
    b.w(contact.inputs['From Min'], 0.0)
    b.l(gin2.outputs['Interact Radius'], contact.inputs['From Max'])
    b.w(contact.inputs['To Min'], 1.0)
    b.w(contact.inputs['To Max'], 0.0)
    contact = contact.outputs['Result']

    depth_neg = b.m('MULTIPLY', gin2.outputs['Interact Depth'], -1.0,
                    -380, -320)
    hn = b.fmix(contact, hn, depth_neg, -200, -100)

    # --- obstacles: pin height to zero (Dirichlet) ---
    co = b.n('GeometryNodeCollectionInfo', -1100, -650,
             transform_space='RELATIVE')
    b.w(co.inputs['Collection'], gin2.outputs['Obstacles'])
    ro = b.n('GeometryNodeRealizeInstances', -920, -650)
    b.l(co.outputs[0], ro.inputs[0])
    prox_o = b.n('GeometryNodeProximity', -740, -650,
                 target_element='FACES')
    b.l(ro.outputs[0], prox_o.inputs[0])
    pin = b.n('ShaderNodeMapRange', -560, -650,
              interpolation_type='SMOOTHSTEP')
    b.w(pin.inputs['Value'], prox_o.outputs['Distance'])
    b.w(pin.inputs['From Min'], 0.0)
    b.l(gin2.outputs['Obstacle Radius'], pin.inputs['From Max'])
    b.w(pin.inputs['To Min'], 1.0)
    b.w(pin.inputs['To Max'], 0.0)
    keep = b.m('SUBTRACT', 1.0, pin.outputs['Result'], -380, -650)
    hn = b.m('MULTIPLY', hn, keep, -40, -20)

    # --- surface vertical velocity (per frame) ---
    velz = b.m('SUBTRACT', hn, h, 120, -160)

    # --- foam potential: aeration proxy = curvature + velocity
    #     fluctuation + direct contact (cf. paper Sec. 5.1) ---
    lap_a  = b.m('ABSOLUTE', lap, x=120, y=-300)
    vel_a  = b.m('ABSOLUTE', velz, x=120, y=-380)
    aer    = b.m('ADD', lap_a, vel_a, 280, -340)
    gen    = b.m('MULTIPLY', aer, gin2.outputs['Foam Gain'], 420, -340)
    gen    = b.m('ADD', gen, contact, 560, -340)
    gen    = b.m('MINIMUM', gen, 1.0, 700, -340)
    f_dec  = b.m('MULTIPLY', foam, gin2.outputs['Foam Decay'], 420, -480)
    foam_n = b.m('MAXIMUM', f_dec, gen, 700, -460)

    # --- wetness: latch where |h| exceeds threshold, slow dry-out ---
    habs  = b.m('ABSOLUTE', hn, x=120, y=-560)
    w_hit = b.n('ShaderNodeMapRange', 280, -560,
                interpolation_type='SMOOTHSTEP')
    b.w(w_hit.inputs['Value'], habs)
    b.w(w_hit.inputs['From Min'], 0.0)
    b.l(gin2.outputs['Wet Sensitivity'], w_hit.inputs['From Max'])
    b.w(w_hit.inputs['To Min'], 0.0)
    b.w(w_hit.inputs['To Max'], 1.0)
    w_dec = b.m('MULTIPLY', wet, gin2.outputs['Wet Decay'], 420, -620)
    wet_n = b.m('MAXIMUM', w_dec, w_hit.outputs['Result'], 700, -600)

    # --- store state ---
    g = b.store(geo, "hp",   h,      'FLOAT', 200, 100)
    g = b.store(g,   "h",    hn,     'FLOAT', 360, 100)
    g = b.store(g,   "velz", velz,   'FLOAT', 520, 100)
    g = b.store(g,   "foam", foam_n, 'FLOAT', 680, 100)
    g = b.store(g,   "wet",  wet_n,  'FLOAT', 840, 100)
    b.l(g, sim_out.inputs['Geometry'])

    # --- post-zone: displace + expose normal for particle guiding ---
    h_out = b.na("h", 'FLOAT', 1050, -200)
    disp  = b.m('MULTIPLY', h_out, gin.outputs['Amplitude'], 1200, -200)
    off   = b.comb(0.0, 0.0, disp, 1350, -200)
    sp = b.n('GeometryNodeSetPosition', 1450, 0)
    b.l(sim_out.outputs['Geometry'], sp.inputs['Geometry'])
    b.l(off, sp.inputs['Offset'])

    nrm = b.n('GeometryNodeInputNormal', 1450, -350)
    g2 = b.store(sp.outputs['Geometry'], "wnorm", nrm.outputs[0],
                 'FLOAT_VECTOR', 1620, 0)
    b.l(g2, gout.inputs['Geometry'])
    return tree


# ------------------------------------------------------------
# Floor wetness sampler tree
# ------------------------------------------------------------

def build_floor_tree():
    tree = _new_tree(FLOOR_TREE)
    b = NT(tree)
    _iface(tree, "Geometry", 'INPUT',  'NodeSocketGeometry')
    _iface(tree, "Geometry", 'OUTPUT', 'NodeSocketGeometry')
    _iface(tree, "Water",    'INPUT',  'NodeSocketObject')

    gin  = b.n('NodeGroupInput', -600, 0)
    gout = b.n('NodeGroupOutput', 400, 0)

    oi = b.n('GeometryNodeObjectInfo', -400, -200,
             transform_space='RELATIVE')
    b.w(oi.inputs['Object'], gin.outputs['Water'])

    p = b.pos(-400, -400)
    wet = b.sns(oi.outputs['Geometry'], "wet", 'FLOAT', p, -100, -300)
    g = b.store(gin.outputs['Geometry'], "wet", wet, 'FLOAT', 100, 0)
    b.l(g, gout.inputs['Geometry'])
    return tree

# ------------------------------------------------------------
# Whitewater particle tree — Weta 2022 adaptation
#   ptype: 0 = bubble, 1 = foam, 2 = spray
# ------------------------------------------------------------

def build_whitewater_tree():
    tree = _new_tree(WW_TREE)
    b = NT(tree)

    _iface(tree, "Geometry", 'INPUT',  'NodeSocketGeometry')
    _iface(tree, "Geometry", 'OUTPUT', 'NodeSocketGeometry')
    _iface(tree, "Water",            'INPUT', 'NodeSocketObject')
    _iface(tree, "Emission Rate",    'INPUT', 'NodeSocketFloat', 400.0, 0.0, 5000.0)
    _iface(tree, "Aeration Min",     'INPUT', 'NodeSocketFloat', 0.25, 0.0, 1.0)
    _iface(tree, "Min Radius",       'INPUT', 'NodeSocketFloat', 0.004, 0.0005, 0.05)
    _iface(tree, "Max Radius",       'INPUT', 'NodeSocketFloat', 0.014, 0.001, 0.1)
    _iface(tree, "Bubble Depth",     'INPUT', 'NodeSocketFloat', 0.15, 0.0, 2.0)
    _iface(tree, "Buoyancy",         'INPUT', 'NodeSocketFloat', 1.2, 0.0, 10.0)
    _iface(tree, "Bubble Drag",      'INPUT', 'NodeSocketFloat', 3.0, 0.0, 20.0)
    _iface(tree, "Guide Strength",   'INPUT', 'NodeSocketFloat', 0.6, 0.0, 5.0)
    _iface(tree, "Surface Drag",     'INPUT', 'NodeSocketFloat', 0.3, 0.0, 5.0)
    _iface(tree, "Cohesion",         'INPUT', 'NodeSocketFloat', 0.4, 0.0, 5.0)
    _iface(tree, "Cohesion Radius",  'INPUT', 'NodeSocketFloat', 0.08, 0.005, 0.5)
    _iface(tree, "Curl",             'INPUT', 'NodeSocketFloat', 0.12, 0.0, 2.0)
    _iface(tree, "Spray Threshold",  'INPUT', 'NodeSocketFloat', 0.012, 0.0, 0.2)
    _iface(tree, "Launch Scale",     'INPUT', 'NodeSocketFloat', 25.0, 0.0, 200.0)
    _iface(tree, "Lifespan Mean",    'INPUT', 'NodeSocketFloat', 2.0, 0.1, 20.0)
    _iface(tree, "Lifespan Spread",  'INPUT', 'NodeSocketFloat', 0.7, 0.0, 10.0)
    _iface(tree, "Wet Life Mult",    'INPUT', 'NodeSocketFloat', 2.0, 0.0, 10.0)
    _iface(tree, "Dry Rate",         'INPUT', 'NodeSocketFloat', 0.99, 0.9, 1.0)
    _iface(tree, "Kill Distance",    'INPUT', 'NodeSocketFloat', 1.5, 0.1, 20.0)
    _iface(tree, "Material",         'INPUT', 'NodeSocketMaterial')

    gin  = b.n('NodeGroupInput', -2600, 400)   # emission params
    gin2 = b.n('NodeGroupInput', -800, -900)   # physics params
    gin3 = b.n('NodeGroupInput', 2400, -400)   # output params
    gout = b.n('NodeGroupOutput', 3200, 0)

    sim_in  = b.n('GeometryNodeSimulationInput', -2300, 0)
    sim_out = b.n('GeometryNodeSimulationOutput', 2400, 0)
    sim_in.pair_with_output(sim_out)
    b.l(gin.outputs['Geometry'], sim_in.inputs['Geometry'])
    dt = sim_in.outputs['Delta Time']

    # evaluated water geometry (with h/foam/wet/velz/wnorm attributes)
    oi = b.n('GeometryNodeObjectInfo', -2600, -400,
             transform_space='RELATIVE')
    b.w(oi.inputs['Object'], gin.outputs['Water'])
    wgeo = oi.outputs['Geometry']

    # ========================= EMISSION =========================
    # density from aeration proxy stored in "foam" on the water mesh
    foam_src = b.na("foam", 'FLOAT', -2400, 500)
    dens = b.n('ShaderNodeMapRange', -2220, 520)
    b.w(dens.inputs['Value'], foam_src)
    b.l(gin.outputs['Aeration Min'], dens.inputs['From Min'])
    b.w(dens.inputs['From Max'], 1.0)
    b.w(dens.inputs['To Min'], 0.0)
    b.l(gin.outputs['Emission Rate'], dens.inputs['To Max'])

    st = b.n('GeometryNodeInputSceneTime', -2400, 700)
    dist = b.n('GeometryNodeDistributePointsOnFaces', -2000, 600,
               distribute_method='RANDOM')
    b.l(wgeo, dist.inputs['Mesh'])
    b.l(dens.outputs['Result'], dist.inputs['Density'])
    b.l(st.outputs['Frame'], dist.inputs['Seed'])
    newp = dist.outputs['Points']

    ppos = b.pos(-1900, 380)
    velz0 = b.sns(wgeo, "velz", 'FLOAT', ppos, -1700, 420)
    wet0  = b.sns(wgeo, "wet",  'FLOAT', ppos, -1700, 300)
    wn0   = b.sns(wgeo, "wnorm", 'FLOAT_VECTOR', ppos, -1700, 180)

    # classification at birth: rising fast -> spray(2);
    # plunging (air entrainment) -> bubble(0); else foam(1)
    is_up   = b.cmp('GREATER_THAN', velz0, gin.outputs['Spray Threshold'],
                    -1500, 460)
    thr_neg = b.m('MULTIPLY', gin.outputs['Spray Threshold'], -1.0, -1560, 360)
    is_dn   = b.cmp('LESS_THAN', velz0, thr_neg, -1400, 380)
    ptype0  = b.sw(is_up, b.sw(is_dn, 1.0, 0.0, -1340, 300), 2.0, -1200, 340)

    # bubble radius from the paper's inverse-cubic CDF (eq. 34):
    #   r = rmin / sqrt(1 - X * (1 - rmin^2/rmax^2))
    X   = b.rand(0.0, 1.0, 11, -1700, 40)
    rr  = b.m('DIVIDE', gin.outputs['Min Radius'], gin.outputs['Max Radius'],
              -1560, 60)
    rr2 = b.m('MULTIPLY', rr, rr, -1420, 60)
    k   = b.m('SUBTRACT', 1.0, rr2, -1290, 60)
    xk  = b.m('MULTIPLY', X, k, -1290, -20)
    den = b.m('SQRT', b.m('SUBTRACT', 1.0, xk, -1160, 0), x=-1040, y=0)
    rad0 = b.m('DIVIDE', gin.outputs['Min Radius'], den, -920, 20)

    # initial velocity: spray launches along surface normal
    is_spray0 = b.cmp('EQUAL', ptype0, 2.0, -1060, 280, eps=0.1)
    launch = b.m('MULTIPLY', velz0, gin.outputs['Launch Scale'], -1060, 200)
    launch = b.m('MULTIPLY', launch, is_spray0, -920, 220)
    vel0 = b.vscale(wn0, launch, -780, 240)

    # bubbles start submerged
    is_bub0 = b.cmp('EQUAL', ptype0, 0.0, -1060, 120, eps=0.1)
    dsub = b.rand(0.2, 1.0, 21, -1060, 60)
    dsub = b.m('MULTIPLY', dsub, gin.outputs['Bubble Depth'], -920, 80)
    dsub = b.m('MULTIPLY', dsub, is_bub0, -790, 100)
    zoff = b.m('MULTIPLY', dsub, -1.0, -670, 100)
    off0 = b.comb(0.0, 0.0, zoff, -560, 120)
    spn = b.n('GeometryNodeSetPosition', -440, 340)
    b.l(newp, spn.inputs['Geometry'])
    b.l(off0, spn.inputs['Offset'])
    newp = spn.outputs['Geometry']

    # per-particle lifespan (uniform approx of the paper's
    # normal-distributed bursting age)
    life0 = b.rand(-1.0, 1.0, 31, -560, -40)
    life0 = b.madd(life0, gin.outputs['Lifespan Spread'],
                   gin.outputs['Lifespan Mean'], -420, -20)

    newp = b.store(newp, "ptype", ptype0, 'FLOAT', -300, 340)
    newp = b.store(newp, "pvel",  vel0,   'FLOAT_VECTOR', -160, 340)
    newp = b.store(newp, "prad",  rad0,   'FLOAT', -20, 340)
    newp = b.store(newp, "page",  0.0,    'FLOAT', 120, 340)
    newp = b.store(newp, "pwet",  1.0,    'FLOAT', 260, 340)
    newp = b.store(newp, "plife", life0,  'FLOAT', 400, 340)

    join = b.n('GeometryNodeJoinGeometry', 560, 100)
    b.l(newp, join.inputs[0])
    b.l(sim_in.outputs['Geometry'], join.inputs[0])
    pts = join.outputs[0]

    # ========================= PHYSICS =========================
    # All new-state fields are captured in the pre-write context,
    # then written, to avoid read-after-write hazards.

    P = b.pos(700, -200)
    v    = b.na("pvel",  'FLOAT_VECTOR', 700, -320)
    ptyp = b.na("ptype", 'FLOAT', 700, -420)
    prad = b.na("prad",  'FLOAT', 700, -500)
    page = b.na("page",  'FLOAT', 700, -580)
    pwet = b.na("pwet",  'FLOAT', 700, -660)
    plif = b.na("plife", 'FLOAT', 700, -740)

    # surface probe (manifold Phi=0 for a heightfield)
    proxw = b.n('GeometryNodeProximity', 900, -200,
                target_element='FACES')
    b.l(wgeo, proxw.inputs[0])
    surf_p  = proxw.outputs['Position']
    surf_d  = proxw.outputs['Distance']
    sx, sy, sz = b.sep(surf_p, 1060, -200)
    px, py, pz = b.sep(P, 1060, -300)

    wn   = b.sns(wgeo, "wnorm", 'FLOAT_VECTOR', P, 900, -420)
    wvz  = b.sns(wgeo, "velz",  'FLOAT', P, 900, -540)
    wwet = b.sns(wgeo, "wet",   'FLOAT', P, 900, -640)

    # guide velocity field u (paper: bulk fluid velocity; here
    # reconstructed from the heightfield: slope + vertical motion)
    wnx, wny, _wnz = b.sep(wn, 1060, -440)
    ux = b.m('MULTIPLY', wnx, gin2.outputs['Guide Strength'], 1200, -420)
    uy = b.m('MULTIPLY', wny, gin2.outputs['Guide Strength'], 1200, -480)
    uz = b.m('MULTIPLY', wvz, 15.0, 1200, -540)   # dh/frame -> m/s ish
    u  = b.comb(ux, uy, uz, 1340, -470)

    # class masks (previous frame's type drives this frame's physics,
    # matching the paper's end-of-step transitions)
    isB = b.cmp('EQUAL', ptyp, 0.0, 900, -760, eps=0.1)
    isF = b.cmp('EQUAL', ptyp, 1.0, 900, -820, eps=0.1)
    isS = b.cmp('EQUAL', ptyp, 2.0, 900, -880, eps=0.1)

    # ---- SPRAY: ballistic ----
    grav = b.vscale((0.0, 0.0, -9.81), dt, 1200, -650)
    vS = b.vm('ADD', v, grav, 1340, -640)

    # ---- BUBBLES (paper Sec. 3): buoyancy ~ r^2 (Stokes regime,
    # big bubbles rise, small ones follow the flow) + linear drag
    # toward the guide velocity ----
    rn  = b.m('DIVIDE', prad, gin2.outputs['Max Radius'], 1200, -760)
    rn2 = b.m('MULTIPLY', rn, rn, 1340, -760)
    bacc = b.m('MULTIPLY', rn2, gin2.outputs['Buoyancy'], 1480, -760)
    bacc = b.m('MULTIPLY', bacc, dt, 1600, -760)
    bvec = b.comb(0.0, 0.0, bacc, 1720, -760)
    vB   = b.vm('ADD', v, bvec, 1840, -720)
    dragB = b.m('MULTIPLY', gin2.outputs['Bubble Drag'], dt, 1600, -840)
    dragB = b.m('MINIMUM', dragB, 1.0, 1720, -840)
    vB    = b.vmix(dragB, vB, u, 1960, -740)

    # ---- FOAM (paper Sec. 6): wet viscous surface fluid ----
    # surface drag a^Phi = chi*(u - v)  (eq. 44)
    dragF = b.m('MULTIPLY', gin2.outputs['Surface Drag'], dt, 1200, -960)
    dragF = b.m('MULTIPLY', dragF, 4.0, 1320, -960)
    dragF = b.m('MINIMUM', dragF, 1.0, 1440, -960)
    vF = b.vmix(dragF, v, u, 1580, -940)

    # cohesion/pressure with effective distance D = |x| - (rp+rq)
    # (paper eq. 43, pairwise via nearest neighbour):
    # attract when separated, repel when overlapping
    ion = b.n('GeometryNodeIndexOfNearest', 1200, -1100)
    b.w(ion.inputs['Position'], P)
    sidx = b.n('GeometryNodeSampleIndex', 1380, -1100,
               data_type='FLOAT_VECTOR', domain='POINT')
    b.l(pts, sidx.inputs['Geometry'])
    b.w(b.find(sidx.inputs, 'Value', 'VECTOR'), P)
    b.l(ion.outputs['Index'], sidx.inputs['Index'])
    npos = b.find(sidx.outputs, 'Value', 'VECTOR')
    dvec = b.vm('SUBTRACT', npos, P, 1560, -1100)
    dlen = b.vm('LENGTH', dvec, x=1700, y=-1100)
    ddir = b.vm('NORMALIZE', dvec, x=1700, y=-1180)
    deff = b.m('SUBTRACT', dlen, b.m('MULTIPLY', prad, 2.0, 1700, -1260),
               1840, -1200)
    dnrm = b.m('DIVIDE', deff, gin2.outputs['Cohesion Radius'], 1960, -1200)
    dnrm = b.m('MAXIMUM', b.m('MINIMUM', dnrm, 1.0, 2080, -1200), -1.0,
               2200, -1200)
    kern = b.n('ShaderNodeMapRange', 1840, -1100,
               interpolation_type='SMOOTHSTEP')
    b.w(kern.inputs['Value'], dlen)
    b.w(kern.inputs['From Min'], 0.0)
    b.l(gin2.outputs['Cohesion Radius'], kern.inputs['From Max'])
    b.w(kern.inputs['To Min'], 1.0)
    b.w(kern.inputs['To Max'], 0.0)
    coh = b.m('MULTIPLY', gin2.outputs['Cohesion'], kern.outputs['Result'],
              2100, -1100)
    coh = b.m('MULTIPLY', coh, dnrm, 2220, -1100)
    coh = b.m('MULTIPLY', coh, ion.outputs['Has Neighbor'], 2340, -1100)
    # wetter foam clumps harder (wetness modulates yield, Sec. 6.1)
    coh = b.m('MULTIPLY', coh, b.m('ADD', 0.3, pwet, 2340, -1180),
              2460, -1120)
    coh = b.m('MULTIPLY', coh, dt, 2580, -1120)
    a_coh = b.vscale(ddir, coh, 2700, -1120)
    vF = b.vm('ADD', vF, a_coh, 2820, -1000)

    # curl-ish surface noise for micro breakup
    nz = b.n('ShaderNodeTexNoise', 1200, -1350, noise_dimensions='4D')
    b.w(nz.inputs['Vector'], P)
    b.w(nz.inputs['Scale'], 3.0)
    b.l(st.outputs['Seconds'], nz.inputs['W'])
    ncen = b.vm('SUBTRACT', nz.outputs['Color'], (0.5, 0.5, 0.5),
                1400, -1350)
    ncrl = b.vscale(ncen, b.m('MULTIPLY', gin2.outputs['Curl'], dt,
                              1400, -1430), 1560, -1360)
    vF = b.vm('ADD', vF, ncrl, 2960, -1000)

    # combine class velocities
    v1 = b.vscale(vS, isS, 2100, -640)
    v2 = b.vscale(vB, isB, 2100, -720)
    v3 = b.vscale(vF, isF, 3080, -1000)
    v_new = b.vm('ADD', b.vm('ADD', v1, v2, 2260, -680), v3, 2400, -820)

    # ---- transitions (paper Algorithm 2) ----
    below = b.cmp('LESS_THAN', pz, sz, 1200, -80)
    near  = b.cmp('LESS_THAN', surf_d,
                  b.m('MULTIPLY', prad, 2.5, 1200, -20), 1340, -40)
    _vx, _vy, vz_c = b.sep(v_new, 2400, -560)
    fastdown = b.cmp('LESS_THAN', vz_c, -0.4, 2540, -520)

    landed   = b.band(isS, below, 1480, -60)
    entrain  = b.band(landed, fastdown, 2680, -420)   # spray -> bubble
    surfaced = b.band(isB, near, 1480, -130)          # bubble -> foam

    toFoam = b.bor(b.band(landed, b.bnot(fastdown, 2680, -350), 2820, -360),
                   surfaced, 2960, -300)
    ptype_n = b.sw(toFoam,
                   b.sw(entrain, ptyp, 0.0, 2960, -220),
                   1.0, 3100, -240)

    # tangential momentum preservation, zeta = 0.7 (paper Sec. 7.1)
    vtan = b.comb(b.m('MULTIPLY', _vx, 0.7, 2540, -600),
                  b.m('MULTIPLY', _vy, 0.7, 2540, -660), 0.0, 2680, -620)
    v_fin = b.vmix(toFoam, v_new, vtan, 3100, -700)

    # ---- integrate positions; manifold constraint for foam ----
    step = b.vscale(v_fin, dt, 3240, -700)
    pos_i = b.vm('ADD', P, step, 3360, -640)
    ix, iy, _iz = b.sep(pos_i, 3480, -640)
    zoffp = b.m('MULTIPLY', prad, 0.6, 3480, -560)
    zsurf = b.m('ADD', sz, zoffp, 3600, -560)
    pos_f = b.comb(ix, iy, zsurf, 3600, -640)
    foam_now = b.bor(isF, toFoam, 3480, -480)
    pos_n = b.vmix(foam_now, pos_i, pos_f, 3740, -600)

    # ---- age / wetness / bursting ----
    age_n = b.m('ADD', page, dt, 1200, 60)
    dry   = b.m('MULTIPLY', pwet, gin2.outputs['Dry Rate'], 1200, 130)
    wet_n = b.m('MAXIMUM', dry, wwet, 1360, 110)
    lifeM = b.madd(wet_n, gin2.outputs['Wet Life Mult'], 1.0, 1360, 30)
    life_e = b.m('MULTIPLY', plif, lifeM, 1520, 40)
    dead  = b.cmp('GREATER_THAN', age_n, life_e, 1660, 40)
    gone  = b.cmp('GREATER_THAN', surf_d, gin2.outputs['Kill Distance'],
                  1660, -20)
    kill  = b.bor(dead, gone, 1800, 20)

    # ---- capture in pre-write context, then write ----
    g, c_vel  = b.capture(pts, v_fin,  'FLOAT_VECTOR', 900, 200)
    g, c_pos  = b.capture(g, pos_n,    'FLOAT_VECTOR', 1060, 200)
    g, c_typ  = b.capture(g, ptype_n,  'FLOAT', 1220, 200)
    g, c_age  = b.capture(g, age_n,    'FLOAT', 1380, 200)
    g, c_wet  = b.capture(g, wet_n,    'FLOAT', 1540, 200)
    g, c_kill = b.capture(g, kill,     'BOOLEAN', 1700, 200)

    spx = b.n('GeometryNodeSetPosition', 1860, 200)
    b.l(g, spx.inputs['Geometry'])
    b.l(c_pos, spx.inputs['Position'])
    g = spx.outputs['Geometry']
    g = b.store(g, "pvel",  c_vel, 'FLOAT_VECTOR', 2000, 200)
    g = b.store(g, "ptype", c_typ, 'FLOAT', 2140, 200)
    g = b.store(g, "page",  c_age, 'FLOAT', 2280, 200)
    g = b.store(g, "pwet",  c_wet, 'FLOAT', 2420, 260)

    dg = b.n('GeometryNodeDeleteGeometry', 2560, 200,
             domain='POINT', mode='ALL')
    b.l(g, dg.inputs['Geometry'])
    b.l(c_kill, dg.inputs['Selection'])
    b.l(dg.outputs['Geometry'], sim_out.inputs['Geometry'])

    # ========================= OUTPUT =========================
    prad_o = b.na("prad", 'FLOAT', 2500, -300)
    ptyp_o = b.na("ptype", 'FLOAT', 2500, -380)
    isF_o = b.cmp('EQUAL', ptyp_o, 1.0, 2650, -380, eps=0.1)
    rmul = b.madd(isF_o, 0.4, 1.0, 2800, -360)     # foam reads bigger
    r_out = b.m('MULTIPLY', prad_o, rmul, 2940, -330)

    spr = b.n('GeometryNodeSetPointRadius', 2800, 0)
    b.l(sim_out.outputs['Geometry'], spr.inputs['Points'])
    b.w(spr.inputs['Radius'], r_out)

    smat = b.n('GeometryNodeSetMaterial', 3000, 0)
    b.l(spr.outputs['Points'], smat.inputs['Geometry'])
    b.w(smat.inputs['Material'], gin3.outputs['Material'])
    b.l(smat.outputs['Geometry'], gout.inputs['Geometry'])
    return tree

# ------------------------------------------------------------
# Materials (read GN attributes directly — no image round-trip)
# ------------------------------------------------------------

def _set_in(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def build_water_material():
    mat = bpy.data.materials.get(WATER_MAT) or bpy.data.materials.new(WATER_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (700, 0)

    att = nt.nodes.new('ShaderNodeAttribute'); att.location = (-500, 100)
    att.attribute_name = "foam"
    att.attribute_type = 'GEOMETRY'

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (-100, 200)
    _set_in(bsdf, ['Base Color'], (0.55, 0.75, 0.85, 1.0))
    _set_in(bsdf, ['IOR'], 1.33)
    _set_in(bsdf, ['Transmission Weight', 'Transmission'], 1.0)

    rough = nt.nodes.new('ShaderNodeMapRange'); rough.location = (-300, 350)
    _set_in(rough, ['From Min'], 0.0); _set_in(rough, ['From Max'], 1.0)
    _set_in(rough, ['To Min'], 0.03);  _set_in(rough, ['To Max'], 0.35)
    nt.links.new(att.outputs['Fac'], rough.inputs['Value'])
    nt.links.new(rough.outputs['Result'], bsdf.inputs['Roughness'])

    foam = nt.nodes.new('ShaderNodeBsdfDiffuse'); foam.location = (-100, -150)
    _set_in(foam, ['Color'], (0.95, 0.97, 1.0, 1.0))

    mix = nt.nodes.new('ShaderNodeMixShader'); mix.location = (350, 50)
    nt.links.new(att.outputs['Fac'], mix.inputs['Fac'])
    nt.links.new(bsdf.outputs['BSDF'], mix.inputs[1])
    nt.links.new(foam.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat


def build_floor_material():
    mat = bpy.data.materials.get(FLOOR_MAT) or bpy.data.materials.new(FLOOR_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (600, 0)

    att = nt.nodes.new('ShaderNodeAttribute'); att.location = (-500, 0)
    att.attribute_name = "wet"
    att.attribute_type = 'GEOMETRY'

    ramp = nt.nodes.new('ShaderNodeValToRGB'); ramp.location = (-250, 150)
    ramp.color_ramp.elements[0].color = (0.45, 0.36, 0.24, 1.0)
    ramp.color_ramp.elements[1].color = (0.12, 0.09, 0.06, 1.0)
    nt.links.new(att.outputs['Fac'], ramp.inputs['Fac'])

    rough = nt.nodes.new('ShaderNodeMapRange'); rough.location = (-250, -180)
    _set_in(rough, ['From Min'], 0.0); _set_in(rough, ['From Max'], 1.0)
    _set_in(rough, ['To Min'], 0.9);   _set_in(rough, ['To Max'], 0.25)
    nt.links.new(att.outputs['Fac'], rough.inputs['Value'])

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (150, 0)
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(rough.outputs['Result'], bsdf.inputs['Roughness'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def build_whitewater_material():
    mat = bpy.data.materials.get(WW_MAT) or bpy.data.materials.new(WW_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (400, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)
    _set_in(bsdf, ['Base Color'], (0.92, 0.95, 1.0, 1.0))
    _set_in(bsdf, ['Roughness'], 0.45)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


# ------------------------------------------------------------
# Scene construction
# ------------------------------------------------------------

def _get_collection(name, scene):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in {c.name for c in scene.collection.children}:
        try:
            scene.collection.children.link(col)
        except Exception:
            pass
    return col


def _mod_set(mod, tree, name, value):
    for sock in tree.interface.items_tree:
        if sock.item_type == 'SOCKET' and sock.in_out == 'INPUT' \
                and sock.name == name:
            mod[sock.identifier] = value
            return True
    return False


class RF_OT_build(bpy.types.Operator):
    """Build the pond, whitewater system, node trees and materials"""
    bl_idname = "ripple_forge.build"
    bl_label = "Build Pond System"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.rf_gn
        scene = context.scene
        size = props.pond_size

        col_i = _get_collection(COL_INTER, scene)
        col_o = _get_collection(COL_OBST, scene)

        water_tree = build_water_tree()
        ww_tree = build_whitewater_tree()
        floor_tree = build_floor_tree()
        wmat = build_water_material()
        fmat = build_floor_material()
        wwmat = build_whitewater_material()

        # --- water surface ---
        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=props.water_subdiv,
            y_subdivisions=props.water_subdiv,
            size=size, location=(0, 0, 0))
        water = context.active_object
        water.name = WATER_OBJ
        bpy.ops.object.shade_smooth()
        water.data.materials.clear()
        water.data.materials.append(wmat)
        mod = water.modifiers.new("RF Water Sim", 'NODES')
        mod.node_group = water_tree
        _mod_set(mod, water_tree, "Interactors", col_i)
        _mod_set(mod, water_tree, "Obstacles", col_o)

        # --- floor ---
        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=64, y_subdivisions=64,
            size=size, location=(0, 0, -props.pond_depth))
        floor = context.active_object
        floor.name = FLOOR_OBJ
        floor.data.materials.clear()
        floor.data.materials.append(fmat)
        fm = floor.modifiers.new("RF Floor Wetness", 'NODES')
        fm.node_group = floor_tree
        _mod_set(fm, floor_tree, "Water", water)

        # --- whitewater host (empty mesh) ---
        me = bpy.data.meshes.new(WW_OBJ)
        ww = bpy.data.objects.new(WW_OBJ, me)
        scene.collection.objects.link(ww)
        wm = ww.modifiers.new("RF Whitewater", 'NODES')
        wm.node_group = ww_tree
        _mod_set(wm, ww_tree, "Water", water)
        _mod_set(wm, ww_tree, "Material", wwmat)

        # --- hidden brush poker (lives in the interactor collection) ---
        if bpy.data.objects.get(BRUSH_OBJ) is None:
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=props.brush_radius, location=(0, 0, -100))
            brush = context.active_object
            brush.name = BRUSH_OBJ
            brush.display_type = 'WIRE'
            brush.hide_render = True
            for c in list(brush.users_collection):
                c.objects.unlink(brush)
            col_i.objects.link(brush)

        scene.frame_start = 1
        scene.frame_set(1)
        v = ".".join(str(i) for i in bl_info["version"])
        self.report({'INFO'},
                    f"Ripple Forge GN {v} — built. Press SPACE to play; all "
                    "parameters live on the two Geometry Nodes modifiers.")
        return {'FINISHED'}

# ------------------------------------------------------------
# Brush: moves the hidden RF_Brush sphere along the mouse ray,
# the GN proximity interaction does the rest (during playback)
# ------------------------------------------------------------

class RF_OT_brush(bpy.types.Operator):
    """Drag on the water during playback to poke ripples (RMB/Esc exits)"""
    bl_idname = "ripple_forge.brush"
    bl_label = "Ripple Brush"

    _dragging = False

    def _park(self):
        brush = bpy.data.objects.get(BRUSH_OBJ)
        if brush:
            brush.location = (0, 0, -100)

    def _move(self, context, event):
        brush = bpy.data.objects.get(BRUSH_OBJ)
        water = bpy.data.objects.get(WATER_OBJ)
        if brush is None or water is None:
            return
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if abs(direction.z) < 1e-6:
            return
        z = water.matrix_world.translation.z
        t = (z - origin.z) / direction.z
        if t < 0:
            return
        p = origin + direction * t
        brush.location = (p.x, p.y, z)

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._park()
            context.window.cursor_modal_restore()
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE':
            self._dragging = (event.value == 'PRESS')
            if self._dragging:
                self._move(context, event)
            else:
                self._park()
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self._dragging:
            self._move(context, event)
            return {'RUNNING_MODAL'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run from the 3D Viewport")
            return {'CANCELLED'}
        if bpy.data.objects.get(WATER_OBJ) is None:
            self.report({'ERROR'}, "Build the pond system first")
            return {'CANCELLED'}
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
        context.window.cursor_modal_set('PAINT_BRUSH')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class RF_OT_setup_caustics(bpy.types.Operator):
    """Configure Cycles MNEE shadow caustics on water, floor and lights"""
    bl_idname = "ripple_forge.setup_caustics"
    bl_label = "Setup Cycles Caustics"

    def execute(self, context):
        water = bpy.data.objects.get(WATER_OBJ)
        floor = bpy.data.objects.get(FLOOR_OBJ)
        if water is None or floor is None:
            self.report({'ERROR'}, "Build the pond system first")
            return {'CANCELLED'}

        def set_flag(target, paths, value):
            for path in paths:
                obj = target
                parts = path.split('.')
                ok = True
                for pp in parts[:-1]:
                    if hasattr(obj, pp):
                        obj = getattr(obj, pp)
                    else:
                        ok = False
                        break
                if ok and hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                    return True
            return False

        set_flag(water, ['is_caustics_caster', 'cycles.is_caustics_caster'], True)
        set_flag(floor, ['is_caustics_receiver', 'cycles.is_caustics_receiver'], True)

        lights = [o for o in context.scene.objects
                  if o.type == 'LIGHT' and o.data.type in {'SUN', 'SPOT', 'POINT'}]
        if not lights:
            bpy.ops.object.light_add(type='SUN', location=(3, -3, 6),
                                     rotation=(math.radians(35), 0,
                                               math.radians(30)))
            sun = context.active_object
            sun.data.energy = 4.0
            lights = [sun]
        n = 0
        for lo in lights:
            if set_flag(lo.data, ['cycles.is_caustics_light',
                                  'use_shadow_caustics'], True):
                n += 1
        context.scene.render.engine = 'CYCLES'
        self.report({'INFO'},
                    f"MNEE caustics on {n} light(s). Unsupported on OptiX — "
                    "use CUDA/HIP/Metal/CPU.")
        return {'FINISHED'}


class RF_OT_bake(bpy.types.Operator):
    """Bake both Simulation Zones to disk cache for deterministic renders"""
    bl_idname = "ripple_forge.bake"
    bl_label = "Bake Simulations"

    def execute(self, context):
        targets = [bpy.data.objects.get(WATER_OBJ),
                   bpy.data.objects.get(WW_OBJ)]
        targets = [t for t in targets if t is not None]
        if not targets:
            self.report({'ERROR'}, "Build the pond system first")
            return {'CANCELLED'}
        for obj in context.selected_objects:
            obj.select_set(False)
        for t in targets:
            t.select_set(True)
        context.view_layer.objects.active = targets[0]
        try:
            bpy.ops.object.simulation_nodes_cache_bake(selected=True)
        except Exception as e:
            self.report({'ERROR'}, f"Bake failed: {e} — you can also bake "
                        "from Properties > Physics > Simulation Nodes")
            return {'CANCELLED'}
        self.report({'INFO'}, "Simulation zones baked")
        return {'FINISHED'}


# ------------------------------------------------------------
# Props + panel + registration
# ------------------------------------------------------------

class RFGNProps(bpy.types.PropertyGroup):
    pond_size: bpy.props.FloatProperty(
        name="Pond Size (m)", default=5.0, min=0.5, max=100.0)
    pond_depth: bpy.props.FloatProperty(
        name="Depth (m)", default=0.5, min=0.05, max=10.0)
    water_subdiv: bpy.props.IntProperty(
        name="Grid Resolution", default=200, min=32, max=400,
        description="Water grid vertices per side (sim resolution)")
    brush_radius: bpy.props.FloatProperty(
        name="Brush Radius (m)", default=0.12, min=0.02, max=1.0)


class RF_PT_panel(bpy.types.Panel):
    bl_label = "Ripple Forge GN"
    bl_idname = "RF_PT_gn_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Ripple Forge"

    def draw(self, context):
        layout = self.layout
        props = context.scene.rf_gn

        box = layout.box()
        box.label(text="Setup", icon='MOD_FLUIDSIM')
        col = box.column(align=True)
        col.prop(props, "pond_size")
        col.prop(props, "pond_depth")
        col.prop(props, "water_subdiv")
        col.prop(props, "brush_radius")
        box.operator("ripple_forge.build", icon='ADD')

        box = layout.box()
        box.label(text="Interact", icon='PLAY')
        row = box.row()
        row.scale_y = 1.3
        row.operator("screen.animation_play",
                     text="Play / Pause Sim", icon='PLAY')
        box.operator("ripple_forge.brush", icon='BRUSH_DATA')
        box.label(text="Drop animated objects into", icon='INFO')
        box.label(text="the RF_Interactors collection.")
        box.label(text="Rocks go in RF_Obstacles.")

        box = layout.box()
        box.label(text="Tune (live)", icon='MODIFIER')
        box.label(text="Water: 'RF Water Sim' modifier")
        box.label(text="Particles: 'RF Whitewater' modifier")

        box = layout.box()
        box.label(text="Render", icon='RESTRICT_RENDER_OFF')
        box.operator("ripple_forge.setup_caustics", icon='LIGHT_SUN')
        box.operator("ripple_forge.bake", icon='RENDER_ANIMATION')


classes = (
    RFGNProps,
    RF_OT_build,
    RF_OT_brush,
    RF_OT_setup_caustics,
    RF_OT_bake,
    RF_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.rf_gn = bpy.props.PointerProperty(type=RFGNProps)


def unregister():
    del bpy.types.Scene.rf_gn
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()

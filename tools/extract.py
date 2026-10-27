"""Extract a 3D-ready model of the typical floor from the AutoCAD PDF plot.

Usage: python -I tools/extract.py <plan.pdf> <out model.js>
Needs: pymupdf, numpy, scipy.

The plot is 1/8" = 1'-0" (9 pt per foot). Layers are told apart by stroke colour:
black = walls, red = columns / lift shaft / beam / stair risers, blue = windows,
orange = doors, brown = balcony rails, dark grey = lift doors, magenta = plot line.
"""
import json
import math
import re
import sys
from collections import defaultdict, deque

import numpy as np
import pymupdf
from scipy import ndimage

PT_PER_FT = 9.0
R = 0.5  # raster cell, pt

BLACK, RED, BLUE, ORANGE = (0, 0, 0), (1, 0, 0), (0, 0.5, 1), (0.58, 0.29, 0)
BROWN, LIFTDOOR, MAGENTA = (0.72, 0.18, 0), (0.33, 0.33, 0.33), (1, 0, 1)


def rnd(t):
    return tuple(round(v, 2) for v in (t or ()))


def load(path):
    page = pymupdf.open(path)[0]
    g = defaultdict(list)
    for d in page.get_drawings():
        g[(rnd(d.get('color') or d.get('fill')), round(d.get('width') or 0, 2))].append(d)
    spans = []
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines', []):
            for s in line['spans']:
                if s['text'].strip():
                    x0, y0, x1, y1 = s['bbox']
                    spans.append((s['text'].strip(), (x0 + x1) / 2, (y0 + y1) / 2))
    return g, spans


def items(ds, kinds):
    return [it for d in ds for it in d['items'] if it[0] in kinds]


def axis_lines(its):
    h, v = [], []
    for it in its:
        a, b = it[1], it[2]
        if abs(a.y - b.y) < 0.1 and abs(a.x - b.x) > 0.1:
            h.append(((a.y + b.y) / 2, min(a.x, b.x), max(a.x, b.x)))
        elif abs(a.x - b.x) < 0.1 and abs(a.y - b.y) > 0.1:
            v.append(((a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)))
    return h, v


def pair_faces(lines, dmin, dmax):
    """Pair each face line with its nearest overlapping parallel line -> solid strip (c0, c1, a, b)."""
    out = []
    for c, a, b in lines:
        cands = [(c2 - c, c2, a2, b2) for c2, a2, b2 in lines
                 if dmin <= c2 - c <= dmax and min(b, b2) - max(a, a2) > 0.3]
        if not cands:
            continue
        best = min(x[0] for x in cands)
        out += [(c, c2, max(a, a2), min(b, b2)) for d, c2, a2, b2 in cands if abs(d - best) < 0.05]
    return out


def strips_to_rects(h, v):
    """Horizontal strips have c along y; vertical strips have c along x. Return (x0,y0,x1,y1)."""
    return [(a, c0, b, c1) for c0, c1, a, b in h] + [(c0, a, c1, b) for c0, c1, a, b in v]


def quad_rect(q):
    xs = [p.x for p in (q.ul, q.ur, q.lr, q.ll)]
    ys = [p.y for p in (q.ul, q.ur, q.lr, q.ll)]
    return min(xs), min(ys), max(xs), max(ys)


def greedy_rects(mask):
    """Merge a boolean grid into axis-aligned rectangles (cell coords, end-exclusive)."""
    done, open_runs = [], {}
    for y in range(mask.shape[0]):
        row = np.concatenate(([0], mask[y].astype(np.int8), [0]))
        d = np.diff(row)
        runs = set(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
        nxt = {}
        for run, y0 in open_runs.items():
            if run in runs:
                nxt[run] = y0
            else:
                done.append((run[0], y0, run[1], y))
        for run in runs:
            nxt.setdefault(run, y)
        open_runs = nxt
    done += [(r[0], y0, r[1], mask.shape[0]) for r, y0 in open_runs.items()]
    return done


def parse_dims(s):
    m = re.match(r"(\d+)'-(\d+)\"X(\d+)'-(\d+)\"", s)
    return (int(m[1]) + int(m[2]) / 12, int(m[3]) + int(m[4]) / 12) if m else None


def main(pdf, out):
    g, spans = load(pdf)
    walls_its = items(g[(BLACK, 0.84)], 'l') + items(g[(RED, 1.14)], 'l')
    plot = [(it[1].x, it[1].y) for it in items(g[(MAGENTA, 1.71)], 'l')]

    # --- grid over the plot line
    X0, Y0 = min(p[0] for p in plot) - 8, min(p[1] for p in plot) - 8
    X1, Y1 = max(p[0] for p in plot) + 8, max(p[1] for p in plot) + 8
    W, H = int((X1 - X0) / R) + 1, int((Y1 - Y0) / R) + 1
    cell = lambda x, y: (int(round((x - X0) / R)), int(round((y - Y0) / R)))

    def grid():
        return np.zeros((H, W), bool)

    def fill(m, r, pad=0.0):
        x0, y0 = cell(r[0] - pad, r[1] - pad)
        x1, y1 = cell(r[2] + pad, r[3] + pad)
        m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True

    # --- walls: paired face lines, then close the T / corner gaps the pairing leaves
    h, v = axis_lines(walls_its)
    wall = grid()
    for r in strips_to_rects(pair_faces(h, 1.5, 9.5), pair_faces(v, 1.5, 9.5)):
        fill(wall, r)
    wall = ndimage.binary_closing(wall, np.ones((9, 9)))

    # --- columns, beam (red outlined quads)
    columns, beams = [], []
    for it in items(g[(RED, 1.14)], ('qu',)):
        r = quad_rect(it[1])
        (columns if max(r[2] - r[0], r[3] - r[1]) < 30 else beams).append(r)
    col = grid()
    for r in columns:
        fill(col, r)

    # --- windows: cluster blue lines
    def clusters(its, dil):
        m = grid()
        for it in its:
            fill(m, (min(it[1].x, it[2].x), min(it[1].y, it[2].y), max(it[1].x, it[2].x), max(it[1].y, it[2].y)))
        lab, n = ndimage.label(ndimage.binary_dilation(m, np.ones((dil, dil))))
        res = []
        for sl in ndimage.find_objects(lab):
            ys, xs = sl
            res.append((X0 + xs.start * R + dil * R / 2, Y0 + ys.start * R + dil * R / 2,
                        X0 + (xs.stop - 1) * R - dil * R / 2, Y0 + (ys.stop - 1) * R - dil * R / 2))
        return res

    def wall_band(r, horizontal):
        """Snap a thin opening rect's depth to the wall run it sits in (probed just past each end)."""
        solid = wall | col
        if horizontal:
            at = lambda along, depth: solid[cell(along, depth)[1], cell(along, depth)[0]]
            a0, a1, c = r[0], r[2], (r[1] + r[3]) / 2
        else:
            at = lambda along, depth: solid[cell(depth, along)[1], cell(depth, along)[0]]
            a0, a1, c = r[1], r[3], (r[0] + r[2]) / 2
        lo, hi = [], []
        for along in (a0 - 1.5, a1 + 1.5):
            start = next((c + o for o in (0, 0.5, -0.5, 1, -1, 1.5, -1.5) if at(along, c + o)), None)
            if start is None:
                continue
            d0 = d1 = start
            while d0 - start > -8 and at(along, d0 - R):
                d0 -= R
            while d1 - start < 8 and at(along, d1 + R):
                d1 += R
            lo.append(d0)
            hi.append(d1)
        d0, d1 = (min(lo), max(hi)) if lo else (c - 1.9, c + 1.9)
        if d1 - d0 > 10:
            d0, d1 = c - 1.9, c + 1.9
        return (a0, d0, a1, d1) if horizontal else (d0, a0, d1, a1)

    windows = []
    for r in clusters(items(g[(BLUE, 0.15)], 'l'), 5):
        horiz = (r[2] - r[0]) > (r[3] - r[1])
        if max(r[2] - r[0], r[3] - r[1]) < 6:
            continue
        windows.append(wall_band(r, horiz))
    win = grid()
    for r in windows:
        fill(win, r)

    # --- lift doors (dark grey), balcony rails (brown)
    # each lift door is drawn as two parallel lines across the shaft wall; merge the pair.
    # The longer pair between the cars is the wall dividing the two shafts.
    grey = [r for r in clusters(items(g[(LIFTDOOR, 0.15)], 'l'), 16) if 3 < min(r[2] - r[0], r[3] - r[1]) < 12]
    lift_doors = [r for r in grey if 18 < max(r[2] - r[0], r[3] - r[1]) < 40]
    for r in grey:
        if max(r[2] - r[0], r[3] - r[1]) >= 40:
            fill(wall, r)
    rh, rv = axis_lines(items(g[(BROWN, 0.42)], 'l'))
    rails = strips_to_rects(pair_faces(rh, 0.4, 3), pair_faces(rv, 0.4, 3))
    rails += [quad_rect(it[1]) for it in items(g[(BROWN, 0.42)], ('qu',))]
    rail = grid()
    for r in rails:
        fill(rail, r)
    ldm = grid()
    for r in lift_doors:
        fill(ldm, r, 0.6)

    # --- doors: leaf (orange 0.36) + swing arc (orange hairline)
    leaf_rects = [r for r in clusters(items(g[(ORANGE, 0.36)], 'l'), 4) if 15 < max(r[2] - r[0], r[3] - r[1]) < 50]
    arc_pts = np.array([(p.x, p.y) for it in items(g[(ORANGE, 0)], 'l')
                        if abs(it[1].x - it[2].x) > 0.05 and abs(it[1].y - it[2].y) > 0.05
                        for p in (it[1], it[2])])
    barrier0 = wall | col
    doors = []
    for r in leaf_rects:
        horiz = (r[2] - r[0]) > (r[3] - r[1])
        if horiz:
            e1, e2 = np.array([r[0], (r[1] + r[3]) / 2]), np.array([r[2], (r[1] + r[3]) / 2])
        else:
            e1, e2 = np.array([(r[0] + r[2]) / 2, r[1]]), np.array([(r[0] + r[2]) / 2, r[3]])
        rad = float(np.linalg.norm(e2 - e1))
        best = None
        for hinge, tip in ((e1, e2), (e2, e1)):
            u = (tip - hinge) / rad
            n = np.array([-u[1], u[0]])
            for s in (1, -1):
                mid = hinge + rad * (u + s * n) / math.sqrt(2)
                score = np.min(np.linalg.norm(arc_pts - mid, axis=1))
                if best is None or score < best[0]:
                    best = (score, hinge, u, s * n)
        score, hinge, u, sn = best
        if score > 3:
            print('door without arc skipped', r)
            continue
        # wall depth: scan behind the hinge (against the leaf direction) just past it along the wall
        probe = hinge + sn * 1.5
        depth = []
        for k in range(0, 30):
            p = probe - u * (k * R) + u * 2.0
            i, j = cell(*p)
            if 0 <= i < W and 0 <= j < H and barrier0[j, i]:
                depth.append(k * R - 2.0)
        if depth:
            d0, d1 = min(depth), max(depth)
        else:
            d0, d1 = 0.0, 3.78
        a = hinge - u * d1
        b = hinge - u * d0 + sn * rad
        opening = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        doors.append(dict(hinge=hinge, u=u, sn=sn, w=rad, opening=opening))

    dm = grid()
    for d in doors:
        fill(dm, d['opening'])

    barrier = wall | col | win | rail | dm | ldm
    for r in beams:
        fill(barrier, r)

    # --- labels
    name_re = re.compile(r'^(BALCONY|TOILET|KITCHEN|C\.BED|M\.BED|DINING|LIVING|LIFT|6\'-0" WIDE PASSAGE|TYPE-[ABC])$')
    names = [s for s in spans if name_re.match(s[0])]
    dims = [s for s in spans if parse_dims(s[0]) or s[0].endswith('sft.')]
    labels = []
    used = set()
    for nm, x, y in names:
        best = min(((math.hypot(x - dx, y - dy), k) for k, (_, dx, dy) in enumerate(dims) if k not in used), default=None)
        dim = None
        if nm != 'BALCONY' and 'PASSAGE' not in nm and best and best[0] < 25:
            used.add(best[1])
            dim = dims[best[1]][0]
            x, y = (x + dims[best[1]][1]) / 2, (y + dims[best[1]][2]) / 2
        labels.append(dict(name=nm, dim=dim, x=x, y=y))

    # stair is labelled LIFT 8'-0"X16'-6" on the drawing
    for l in labels:
        if l['name'] == 'LIFT' and l['dim'] and l['dim'].startswith("8'"):
            l['name'] = 'STAIR'
        if 'PASSAGE' in l['name']:
            l['name'] = 'PASSAGE'

    # --- flood fill rooms
    # seal hairline junction gaps (< ~16 in) for the flood fill only; doors and passages are far wider
    lab, n = ndimage.label(~ndimage.binary_closing(barrier, np.ones((24, 24))))
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    if len(sys.argv) > 3:
        from PIL import Image
        img = np.full((H, W, 3), 255, np.uint8)
        img[np.isin(lab, list(border))] = (200, 230, 255)
        img[wall] = (30, 30, 30); img[col] = (200, 0, 0); img[win] = (0, 120, 255)
        img[dm] = (255, 150, 0); img[rail] = (140, 60, 20); img[ldm] = (0, 160, 0)
        Image.fromarray(img).save(sys.argv[3].replace('.png', '_barrier.png'))

    def region_at(x, y):
        """Most common room region around a label (labels can sit on a beam or wall edge)."""
        i, j = cell(x, y)
        for rr in range(0, 20):
            w_ = lab[max(j - rr, 0):j + rr + 1, max(i - rr, 0):i + rr + 1]
            if any(v and v not in border for v in np.unique(w_)):
                rr += 8
                w_ = lab[max(j - rr, 0):j + rr + 1, max(i - rr, 0):i + rr + 1].ravel()
                w_ = w_[(w_ != 0) & ~np.isin(w_, list(border))]
                return int(np.bincount(w_).argmax())
        return None

    for l in labels:
        l['region'] = region_at(l['x'], l['y'])
    types = {l['name'][-1]: l for l in labels if l['name'].startswith('TYPE')}
    roomlabels = [l for l in labels if not l['name'].startswith('TYPE')]
    region_ids = sorted({l['region'] for l in roomlabels if l['region']})
    sizes = np.bincount(lab.ravel())
    for rid in range(1, n + 1):
        sft = sizes[rid] * (R / PT_PER_FT) ** 2
        if rid not in border and rid not in region_ids and 20 < sft < 60:
            js, is_ = np.nonzero(lab == rid)
            roomlabels.append(dict(name='TOILET', dim=None, inferred=True, region=rid,
                                   x=X0 + is_.mean() * R, y=Y0 + js.mean() * R))
            region_ids.append(rid)

    # split shared regions (open living/dining) by nearest label
    room_mask = {}
    yy, xx = np.mgrid[0:H, 0:W]
    for rid in region_ids:
        ls = [k for k, l in enumerate(roomlabels) if l['region'] == rid]
        m = lab == rid
        if len(ls) == 1:
            room_mask[ls[0]] = m
            continue
        ds = np.stack([np.hypot(xx - cell(roomlabels[k]['x'], 0)[0], yy - cell(0, roomlabels[k]['y'])[1]) for k in ls])
        near = np.argmin(ds, axis=0)
        for idx, k in enumerate(ls):
            room_mask[k] = m & (near == idx)

    # --- units: graph over regions through door openings
    edges = defaultdict(set)
    for d in doors:
        c = np.array([(d['opening'][0] + d['opening'][2]) / 2, (d['opening'][1] + d['opening'][3]) / 2])
        span = max(d['opening'][2] - d['opening'][0], d['opening'][3] - d['opening'][1])
        rs = set()
        for s in (1, -1):
            for k in range(2, 16):
                p = c + d['u'] * s * k
                i, j = cell(*p)
                if lab[j, i] and lab[j, i] not in border:
                    rs.add(int(lab[j, i]))
                    break
        d['regions'] = rs
        for a in rs:
            for b in rs:
                if a != b:
                    edges[a].add(b)
    common = {l['region'] for l in roomlabels if l['name'] in ('PASSAGE', 'STAIR', 'LIFT')}
    unit_of = {}
    for t, l in types.items():
        q = deque([l['region']])
        while q:
            r_ = q.popleft()
            if r_ in unit_of or r_ in common:
                continue
            unit_of[r_] = t
            q.extend(edges[r_])
    # rooms with no door into a unit (balconies off windows): nearest unit by touching cells
    for rid in region_ids:
        if rid in unit_of or rid in common:
            continue
        grown = ndimage.binary_dilation(lab == rid, np.ones((41, 41)))
        votes = defaultdict(int)
        for r_, t in unit_of.items():
            if (c_ := int((grown & (lab == r_)).sum())):
                votes[t] += c_
        if votes:
            unit_of[rid] = max(votes, key=votes.get)

    # --- world transform: north (-x on the sheet) -> -Z, screen right -> +X
    foot_lab, _ = ndimage.label(~np.isin(lab, list(border)))
    foot = foot_lab == np.argmax(np.bincount(foot_lab.ravel())[1:]) + 1
    wall &= foot
    col &= foot
    ys_, xs_ = np.nonzero(foot)
    cx, cy = X0 + (xs_.min() + xs_.max()) / 2 * R, Y0 + (ys_.min() + ys_.max()) / 2 * R

    def P(x, y):
        return [round((cy - y) / PT_PER_FT, 3), round((x - cx) / PT_PER_FT, 3)]

    def Rr(r):
        a, b = P(r[0], r[1]), P(r[2], r[3])
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]

    def V(vec):
        return [round(-vec[1], 4), round(vec[0], 4)]

    def mask_rects(m):
        return [Rr((X0 + x0 * R - R / 2, Y0 + y0 * R - R / 2, X0 + x1 * R - R / 2, Y0 + y1 * R - R / 2))
                for x0, y0, x1, y1 in greedy_rects(m)]

    kinds = {'C.BED': 'bed', 'M.BED': 'bed', 'TOILET': 'bath', 'KITCHEN': 'kitchen', 'DINING': 'dining',
             'LIVING': 'living', 'BALCONY': 'balcony', 'LIFT': 'lift', 'STAIR': 'stair', 'PASSAGE': 'lobby'}
    pretty = {'C.BED': 'Bedroom', 'M.BED': 'Master Bedroom', 'TOILET': 'Toilet', 'KITCHEN': 'Kitchen',
              'DINING': 'Dining', 'LIVING': 'Living', 'BALCONY': 'Balcony', 'LIFT': 'Lift', 'STAIR': 'Staircase',
              'PASSAGE': 'Common Passage'}
    rooms = []
    for k, l in enumerate(roomlabels):
        if k not in room_mask:
            print('label without room', l['name'], l['dim'])
            continue
        m = room_mask[k]
        area = m.sum() * (R / PT_PER_FT) ** 2
        rooms.append(dict(name=pretty[l['name']], code=l['name'], inferred=l.get('inferred', False), kind=kinds[l['name']], dim=l['dim'],
                          unit=unit_of.get(l['region']), area=round(float(area), 1), at=P(l['x'], l['y']),
                          rects=mask_rects(m)))

    # --- stair: risers (red hairline), two flights split at the handrail gap
    sh, _ = axis_lines(items(g[(RED, 0.57)], 'l'))
    xs_split = sorted({(round(a, 1), round(b, 1)) for c, a, b in sh})
    flights = []
    for a, b in xs_split:
        ys = sorted(c for c, a2, b2 in sh if round(a2, 1) == a and round(b2, 1) == b)
        flights.append(dict(x=[a, b], y=ys))
    stair_room = next(r for r in rooms if r['kind'] == 'stair')
    stair = dict(flights=[dict(a=P(f['x'][0], f['y'][0]), b=P(f['x'][1], f['y'][-1]), n=len(f['y'])) for f in flights],
                 risers=[[P(f['x'][0], y)[0] for y in f['y']] for f in flights])

    tables = [quad_rect(it[1]) for it in items(g[(BLACK, 0.15)], ('qu',))]

    # kitchen counters: drawn as a thin L of front edges; the counter fills back to the kitchen wall
    kitchens = []
    for k, l in enumerate(roomlabels):
        if l['name'] == 'KITCHEN' and k in room_mask:
            js, is_ = np.nonzero(room_mask[k])
            kitchens.append((X0 + is_.min() * R, Y0 + js.min() * R, X0 + is_.max() * R, Y0 + js.max() * R))
    ch, cv = axis_lines(items(g[(BLACK, 0.15)], 'l'))
    counters = []
    for horizontal, lines in ((True, ch), (False, cv)):
        for c, a, b in lines:
            for kb in kitchens:
                lo, hi = (kb[1], kb[3]) if horizontal else (kb[0], kb[2])
                a0, a1 = (kb[0], kb[2]) if horizontal else (kb[1], kb[3])
                if not (a0 - 2 <= a and b <= a1 + 2 and lo < c < hi):
                    continue
                d0, d1 = (lo, c) if c - lo < hi - c else (c, hi)
                if d1 - d0 < 30:
                    counters.append((a, d0, b, d1) if horizontal else (d0, a, d1, b))

    balcony_regions = {l['region'] for l in roomlabels if l['name'] == 'BALCONY'}

    def is_slider(w):
        # the drawing gives balconies no swing doors: the glazing between a room and its balcony is the way out
        horiz = (w[2] - w[0]) > (w[3] - w[1])
        c = ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        for s_ in (1, -1):
            for k in range(2, 16):
                p = (c[0], c[1] + s_ * k) if horiz else (c[0] + s_ * k, c[1])
                i, j = cell(*p)
                if lab[j, i]:
                    if lab[j, i] in balcony_regions:
                        return True
                    break
        return False

    model = dict(
        source='jannatul_mawa_building_arch.pdf, sheet A-02 Typical Floor Plan',
        walls=mask_rects(wall & ~win & ~dm),
        columns=[Rr(r) for r in columns],
        beams=[Rr(r) for r in beams],
        windows=[dict(r=Rr(w), slider=is_slider(w)) for w in windows],
        rails=mask_rects(rail & ~wall),
        liftDoors=[Rr(r) for r in lift_doors],
        doors=[dict(h=P(*d['hinge']), u=V(d['u']), s=V(d['sn']), w=round(d['w'] / PT_PER_FT, 3),
                    o=Rr(d['opening'])) for d in doors],
        tables=[Rr(r) for r in tables],
        counters=[Rr(r) for r in counters],
        stair=stair,
        rooms=rooms,
        units={t: dict(sft=int(re.sub(r'\D', '', next(s[0] for s in spans if s[0].endswith('sft.') and
                                                       math.hypot(s[1] - l['x'], s[2] - l['y']) < 30))))
               for t, l in types.items()},
        footprint=mask_rects(foot),
        perimeter=mask_rects(foot & ~ndimage.binary_erosion(foot, np.ones((9, 9)))),
        plot=[P(x, y) for x, y in plot],
    )

    # --- self-checks
    # every drawn wall line must end up inside some modelled element; report the ones that do not
    built = ndimage.binary_dilation(wall | col | win | dm | rail | ldm | barrier, np.ones((5, 5)))
    loose = []
    for it in walls_its:
        a, b = it[1], it[2]
        n_ = max(2, int(math.hypot(b.x - a.x, b.y - a.y) / R))
        pts = [(a.x + (b.x - a.x) * t / n_, a.y + (b.y - a.y) * t / n_) for t in range(n_ + 1)]
        miss = [p for p in pts if not built[min(max(cell(*p)[1], 0), H - 1), min(max(cell(*p)[0], 0), W - 1)]]
        if len(miss) * R > 2:
            loose.append((round(len(miss) * R / PT_PER_FT, 2), P(*miss[0]), P(*miss[-1])))
    print('drawn wall lines not covered by the model (ft, from, to):', len(loose))
    for x in sorted(loose, reverse=True):
        print('  ', x)
    assert len(types) == 3 and all(r['unit'] or r['kind'] in ('lift', 'stair', 'lobby') for r in rooms), \
        [(r['name'], r['unit']) for r in rooms]
    for r in rooms:
        d = parse_dims(r['dim'] or '')
        if d and r['kind'] not in ('lift',):
            print(f"{r['name']:16} {r['unit']} {r['dim']:14} drawn {d[0] * d[1]:6.1f} sft  model {r['area']:6.1f} sft")
    for t in sorted(types):
        print('unit', t, 'rooms', sum(r['unit'] == t for r in rooms), 'sum area', round(sum(r['area'] for r in rooms if r['unit'] == t)))
    print('counters', len(counters), [[round(v) for v in Rr(r)] for r in counters])
    print('sliders', sum(w['slider'] for w in model['windows']), 'balconies', len(balcony_regions))
    print('walls', len(model['walls']), 'doors', len(doors), 'windows', len(windows), 'columns', len(columns),
          'rails', len(model['rails']), 'liftDoors', len(lift_doors), 'flights', [f['n'] for f in stair['flights']])

    with open(out, 'w') as f:
        f.write('// Generated by tools/extract.py from the architectural PDF. Do not edit by hand.\n')
        f.write('window.MODEL = ' + json.dumps(model, separators=(',', ':')) + ';\n')

    # debug raster
    if len(sys.argv) > 3:
        from PIL import Image
        img = np.full((H, W, 3), 255, np.uint8)
        pal = {'A': (214, 196, 160), 'B': (170, 200, 190), 'C': (200, 180, 200), None: (220, 220, 220)}
        for k, l in enumerate(roomlabels):
            if k in room_mask:
                img[room_mask[k]] = pal.get(unit_of.get(l['region']), (230, 230, 120))
        img[wall] = (30, 30, 30)
        img[col] = (200, 0, 0)
        img[win] = (0, 120, 255)
        img[dm] = (255, 150, 0)
        img[rail] = (140, 60, 20)
        img[ldm] = (0, 160, 0)
        Image.fromarray(img).save(sys.argv[3])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

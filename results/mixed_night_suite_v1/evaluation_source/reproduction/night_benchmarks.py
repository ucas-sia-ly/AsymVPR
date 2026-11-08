"""Night-query evaluation protocols; independent of training experiments.

MSLS label subsets and RobotCar/DarkDriving diagnostics are explicitly named
custom protocols. No unverified frame correspondence is used as ground truth.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from reproduction.benchmarks import Benchmark, checked_positives, sha256

ROOT = Path('/home/admin123/github/datasets')


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def flag(value):
    if str(value).lower() not in ('true', 'false'):
        raise ValueError(f'Invalid boolean: {value}')
    return str(value).lower() == 'true'


def positives(refcoords, qcoords):
    r, q = np.asarray(refcoords, float), np.asarray(qcoords, float)
    if not np.isfinite(r).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite coordinates')
    return [np.asarray(sorted(x), np.int64) for x in cKDTree(r).query_ball_point(q, 25.)]


def msls(split):
    root = ROOT / f'msls_{split}'
    cities = ['cph', 'sf'] if split == 'val' else ['miami', 'athens', 'buenosaires', 'stockholm', 'bengaluru', 'kampala']
    refs, queries, gt, sources = [], [], [], {}
    excluded = 0
    for city in cities:
        selected = {}
        for role in ('database', 'query'):
            base = root / city / role
            post, raw = rows(base / 'postprocessed.csv'), rows(base / 'raw.csv')
            for fn in ('postprocessed.csv', 'raw.csv'):
                sources[str(base / fn)] = sha256(base / fn)
            if len(post) != len(raw) or any(a['key'] != b['key'] for a,b in zip(post,raw)):
                raise ValueError(f'MSLS metadata order mismatch {base}')
            selected[role] = [a for a,b in zip(post,raw) if not flag(b['pano']) and
                              (flag(a['night']) if role == 'query' else not flag(a['night']))]
        db, qs = selected['database'], selected['query']
        offset = len(refs)
        refs.extend(root / city / 'database/images' / (x['key'] + '.jpg') for x in db)
        if not qs:
            continue
        ps = positives([[x['easting'], x['northing']] for x in db],
                       [[x['easting'], x['northing']] for x in qs])
        for q,p in zip(qs,ps):
            if not len(p):
                excluded += 1
                continue
            queries.append(root / city / 'query/images' / (q['key'] + '.jpg'))
            gt.append(p + offset)
    return Benchmark(f'msls_{split}_night_day',root,refs,queries,checked_positives(gt,len(refs),len(queries)),
        dict(type='custom night-label subset; not official all/subtask benchmark',
             query='all postprocessed night=True, raw pano=False',gallery='all night=False, pano=False',
             positive='same-city UTM Euclidean <=25m',excluded_no_positive=excluded),sources)


def svox():
    root = ROOT / 'svox/images/test'
    refs, queries = sorted((root/'gallery').glob('*.jpg')), sorted((root/'queries_night').glob('*.jpg'))
    def xy(p):
        fields = p.name.split('@')
        return [float(fields[1]),float(fields[2])]
    gt = positives([xy(p) for p in refs],[xy(p) for p in queries])
    keep = [i for i,p in enumerate(gt) if len(p)]
    return Benchmark('svox_test_night',root,refs,[queries[i] for i in keep],[gt[i] for i in keep],
        dict(type='test night queries',positive='UTM Euclidean <=25m',
             queries_before_gt_filter=len(queries),excluded_no_positive=len(queries)-len(keep)),{})


def darkdriving():
    root = ROOT / 'DarkDriving_Base'
    day = {p.name:p for p in (root/'Day').iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')}
    night = {p.name:p for p in (root/'Night').iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')}
    if day.keys() != night.keys() or not day:
        raise ValueError('DarkDriving pair keys mismatch')
    keys = sorted(day)
    return Benchmark('darkdriving_paired_diagnostic',root,[day[k] for k in keys],[night[k] for k in keys],
        [np.asarray([i],np.int64) for i in range(len(keys))],
        dict(type='strict exact-pair retrieval diagnostic, NOT spatial VPR Recall',
             positive='identical Day/Night basename only',split='all locally cleaned pairs; no training on this dataset',
             limitation='nearby frames may be spatial positives but count as negatives'),{})


def robotcar():
    root = ROOT / 'RobotCar'
    days = ['2015-02-06-13-57-16','2014-07-14-14-49-50','2015-08-12-15-04-18']
    nights = ['2014-11-14-16-34-33','2014-12-16-18-44-24','2014-12-10-18-10-50']
    refs, rc, queries, qc, sources = [], [], [], [], {}
    missing = 0
    for traversal in days+nights:
        base = root/('day' if traversal in days else 'night')/traversal
        source = base/'gps/gps_sampled_1m.csv'
        sources[str(source)] = sha256(source)
        seen = set()
        for row in rows(source):
            p = base/'stereo/centre'/(row['timestamp']+'.png')
            if p in seen:
                raise ValueError('Duplicate GPS timestamp')
            seen.add(p)
            if not p.is_file():
                missing += 1
                continue
            if row['utm_zone'] != '30U':
                raise ValueError('Unexpected RobotCar UTM zone')
            if traversal in days:
                refs.append(p); rc.append([row['easting'],row['northing']])
            else:
                queries.append(p); qc.append([row['easting'],row['northing']])
    if not refs or not queries:
        raise ValueError('No exact sampled GPS/image matches')
    gt = positives(rc,qc)
    keep = [i for i,p in enumerate(gt) if len(p)]
    return Benchmark('robotcar_gps_night_diagnostic',root,refs,[queries[i] for i in keep],[gt[i] for i in keep],
        dict(type='local GPS retrieval diagnostic, NOT RobotCar-Seasons localization protocol',
             day_traversals=days,night_traversals=nights,positive='UTM Euclidean <=25m, no heading filter',
             sampling='existing gps_sampled_1m.csv; exact timestamp image join',
             missing_images=missing,excluded_no_positive=len(queries)-len(keep)),sources)


BUILDERS = {'msls_val':lambda:msls('val'),'msls_test':lambda:msls('test'),
            'svox':svox,'darkdriving':darkdriving,'robotcar':robotcar}


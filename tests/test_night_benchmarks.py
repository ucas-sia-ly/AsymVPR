import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from reproduction import night_benchmarks as suite


class NightProtocolTests(unittest.TestCase):
    def test_radius_includes_boundary(self):
        got = suite.positives([[0,0],[25,0],[25.0001,0]],[[0,0]])
        np.testing.assert_array_equal(got[0],[0,1])

    def test_msls_role_labels_and_city_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for city in ['cph','sf']:
                for role in ['database','query']:
                    base = root/'msls_val'/city/role
                    base.mkdir(parents=True)
                    entries = [dict(key='day',easting=0,northing=0,night=False,pano=False),
                               dict(key='night',easting=0,northing=0,night=True,pano=False),
                               dict(key='pano',easting=0,northing=0,night=True,pano=True)]
                    with open(base/'postprocessed.csv','w') as f:
                        w=csv.DictWriter(f,fieldnames=['key','easting','northing','night'])
                        w.writeheader(); w.writerows({k:r[k] for k in w.fieldnames} for r in entries)
                    with open(base/'raw.csv','w') as f:
                        w=csv.DictWriter(f,fieldnames=['key','pano'])
                        w.writeheader(); w.writerows({k:r[k] for k in w.fieldnames} for r in entries)
            with patch.object(suite,'ROOT',root):
                b=suite.msls('val')
            self.assertEqual(len(b.references),2)
            self.assertEqual(len(b.queries),2)
            self.assertTrue(all(p.name=='day.jpg' for p in b.references))
            self.assertTrue(all(p.name=='night.jpg' for p in b.queries))
            np.testing.assert_array_equal(b.positives[0],[0])
            np.testing.assert_array_equal(b.positives[1],[1])


if __name__=='__main__':
    unittest.main()

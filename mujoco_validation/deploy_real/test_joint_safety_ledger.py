import json
from pathlib import Path
import tempfile
import unittest
from generate_joint_safety import generate
from harden_native_release import harden
from finalize_joint_guard_release import finalize


class LedgerGuards(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((Path(__file__).parent/'joint_safety_limits.json').read_text())

    def test_bad_intervals_or_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for key,value in [('front_left_lower_rad',[3,0,0]),('leg_velocity_rad_s',[45,-1,22]),
                              ('user_upper_rad',[float('nan')]*12),('incident_segment_bytes',-1)]:
                data=dict(self.data);data[key]=value;(root/'ledger.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):generate(root/'ledger.json',root/'compiled.hpp')
                self.assertFalse((root/'compiled.hpp').exists())

    def test_only_knees_use_confirmed_operator_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ledger.json';header=root/'compiled.hpp'
            arrays=[]
            for policy in ('sdk_intersection','operator_confirmed'):
                data=dict(self.data,knee_limit_source=policy,position_limit_source='sdk_intersection',
                          front_left_lower_rad=[-.4363,-2.443,-2.758],
                          front_left_upper_rad=[.6109,2.443,2.758])
                source.write_text(json.dumps(data));generate(source,header)
                arrays.append({key:[float(v.rstrip('f')) for v in header.read_text().split(key+' = {')[1].split('}')[0].split(',')]
                               for key in ('effective_lower','effective_upper')})
            for key,user_key in (('effective_lower','user_lower_rad'),('effective_upper','user_upper_rad')):
                for joint in range(12):
                    self.assertEqual(arrays[1][key][joint],self.data[user_key][joint] if joint%3==2 else arrays[0][key][joint])
            data['operator_limits_confirmed']=False
            source.write_text(json.dumps(data))
            with self.assertRaises(ValueError):generate(source,header)

    def test_named_reference_bounds_preserve_front_rear_asymmetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ledger.json';header=root/'compiled.hpp'
            data=dict(self.data,position_limit_source='reference_software')
            # Generic FL bounds must not narrow the explicitly named envelopes.
            data.update(front_left_lower_rad=[-.1,-.2,-.3],front_left_upper_rad=[.1,.2,.3])
            source.write_text(json.dumps(data));generate(source,header)
            for key,expected in [('effective_lower',data['user_lower_rad']),('effective_upper',data['user_upper_rad'])]:
                actual=[float(v.rstrip('f')) for v in header.read_text().split(key+' = {')[1].split('}')[0].split(',')]
                self.assertEqual(actual,expected)

    def test_tracking_override_requires_named_joint_and_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ledger.json';header=root/'compiled.hpp'
            for override,confirmed in [({'wrong_joint':{'maximum_error_rad':.44,'reference_position_gain':80}},True),
                                       ({'hr_hipx_joint':{'maximum_error_rad':.44,'reference_position_gain':80}},False),
                                       ({'hr_hipx_joint':{'maximum_error_rad':float('nan'),'reference_position_gain':80}},True)]:
                data=dict(self.data,tracking_error_overrides=override,tracking_error_overrides_confirmed=confirmed)
                source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):generate(source,header)
                self.assertFalse(header.exists())

    def test_named_tracking_overrides_map_to_sdk_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ledger.json';header=root/'compiled.hpp'
            source.write_text(json.dumps(self.data));generate(source,header)
            text=header.read_text()
            for key in ('tracking_error_max_rad','tracking_error_proxy_nm'):
                values=[float(v) for v in text.split(key+' = {')[1].split('}')[0].split(',')]
                expected=[]
                for i in range(16):
                    if key=='tracking_error_max_rad':
                        expected.append(.65 if i%4==0 else 0.0)
                    elif i%4==0:
                        expected.append(52.0)
                    elif i%4==1:
                        expected.append(95.0)
                    elif i%4==2:
                        expected.append(85.0)
                    else:
                        expected.append(0.0)
                self.assertEqual(values,expected)

    def test_unknown_baseline_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sums').write_text('unreviewed manifest')
            with self.assertRaisesRegex(ValueError,'Not the audited'):harden(root,root/'sums')
            self.assertEqual(sorted(p.name for p in root.iterdir()),['sums'])

    def test_finalization_wrong_board_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                finalize(root/'new',root/'old','00:00:00:00:00:01','wrong-board-fixture')
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()

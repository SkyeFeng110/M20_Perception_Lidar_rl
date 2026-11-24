"""Generate compiled constants from the one reviewed safety ledger."""
import hashlib
import json
import math
from pathlib import Path


def generate(path, output):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data['schema'] != 1 or type(data['operator_limits_confirmed']) is not bool:
        raise ValueError('Invalid safety ledger')
    position_source=data.get('position_limit_source','sdk_intersection')
    if position_source not in ('sdk_intersection','reference_software'):
        raise ValueError('Invalid position limit source')
    if position_source=='reference_software' and not data['operator_limits_confirmed']:
        raise ValueError('Reference software limits require confirmed bounds')
    knee_source=data.get('knee_limit_source','sdk_intersection')
    if knee_source not in ('sdk_intersection','operator_confirmed') or (
            knee_source=='operator_confirmed' and not data['operator_limits_confirmed']):
        raise ValueError('Invalid or unconfirmed knee limit source')
    lo, hi = data['front_left_lower_rad'], data['front_left_upper_rad']
    for key in ('front_left_lower_rad', 'front_left_upper_rad', 'leg_velocity_rad_s', 'leg_torque_nm'):
        v = data[key]
        if len(v) != 3 or not all(type(x) in (float, int) and math.isfinite(x) for x in v):
            raise ValueError(key)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError('Empty angle interval')
    effective_lo, effective_hi = [], []
    for key in ('user_lower_rad', 'user_upper_rad'):
        if len(data[key]) != 12 or not all(type(x) in (float, int) and math.isfinite(x) for x in data[key]):
            raise ValueError(key)
    for j in range(12):
        leg, axis = divmod(j, 3)
        sdk_lo = -hi[0] if leg % 2 and axis == 0 else lo[axis]
        sdk_hi = -lo[0] if leg % 2 and axis == 0 else hi[axis]
        direct=position_source=='reference_software' or (axis==2 and knee_source=='operator_confirmed')
        effective_lo.append(data['user_lower_rad'][j] if direct
                            else max(sdk_lo, data['user_lower_rad'][j]))
        effective_hi.append(data['user_upper_rad'][j] if direct
                            else min(sdk_hi, data['user_upper_rad'][j]))
        if effective_lo[-1] >= effective_hi[-1]:
            raise ValueError('No common safe interval')
    target_rate=data.get('target_rate_rad_s',data['leg_velocity_rad_s'])
    if len(target_rate) != 3 or not all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in target_rate):
        raise ValueError('target_rate_rad_s')
    for key in ('leg_velocity_rad_s', 'leg_torque_nm'):
        if any(v <= 0 for v in data[key]):
            raise ValueError(key)
    names=[f'{leg}_{axis}_joint' for leg in ('fl','fr','hl','hr') for axis in ('hipx','hipy','knee')]
    tracking_rad=[0.0]*16
    tracking_proxy=[data['leg_torque_nm'][i%4] if i%4!=3 else 0.0 for i in range(16)]
    overrides=data.get('tracking_error_overrides',{})
    if not isinstance(overrides,dict) or (overrides and data.get('tracking_error_overrides_confirmed') is not True):
        raise ValueError('Tracking error overrides require explicit confirmation')
    for name,override in overrides.items():
        if name not in names or not isinstance(override,dict):raise ValueError('Invalid tracking override joint')
        error=override.get('maximum_error_rad');gain=override.get('reference_position_gain')
        if any(type(x) not in (float,int) or not math.isfinite(x) or x<=0 for x in (error,gain)):
            raise ValueError('Invalid tracking override value')
        index=names.index(name);sdk=(index//3)*4+index%3
        if error>effective_hi[index]-effective_lo[index]:raise ValueError('Tracking override exceeds entire joint interval')
        tracking_rad[sdk]=error
        tracking_proxy[sdk]=error*gain
    proxy_overrides=data.get('tracking_error_proxy_overrides',{})
    if not isinstance(proxy_overrides,dict) or (
            proxy_overrides and data.get('tracking_error_proxy_overrides_confirmed') is not True):
        raise ValueError('Tracking proxy overrides require explicit confirmation')
    for name,proxy in proxy_overrides.items():
        if name not in names or not isinstance(proxy,(int,float)) or isinstance(proxy,bool) or not math.isfinite(proxy) or proxy<=0:
            raise ValueError('Invalid tracking proxy override')
        index=names.index(name);sdk=(index//3)*4+index%3
        if index%3==3: raise ValueError('Wheel tracking proxy override')
        tracking_proxy[sdk]=proxy
    positive = ('control_tick_s', 'policy_period_s', 'feedback_timeout_s', 'incident_segment_bytes',
                'incident_segments', 'incident_disk_reserve_bytes', 'incident_sync_s', 'incident_post_fault_s', 'incident_queue_capacity')
    for key in positive:
        if type(data[key]) not in (float, int) or not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(key)
    for key in ('incident_segment_bytes', 'incident_segments', 'incident_disk_reserve_bytes', 'incident_queue_capacity'):
        if type(data[key]) is not int:
            raise ValueError(key)
    if not data['control_tick_s'] <= data['policy_period_s'] < data['feedback_timeout_s']:
        raise ValueError('Invalid timing order')
    lines = ['#pragma once', '#include <array>', '#include <cstdint>', 'namespace m20 { namespace joint_limits {',
             'inline constexpr const char* ledger_sha256 = "'+hashlib.sha256(raw).hexdigest()+'";',
             'inline constexpr bool operator_limits_confirmed = '+str(data['operator_limits_confirmed']).lower()+';']
    for key in ('front_left_lower_rad', 'front_left_upper_rad', 'leg_velocity_rad_s', 'leg_torque_nm'):
        lines.append('inline constexpr std::array<double,3> '+key+' = {'+','.join(repr(v) for v in data[key])+'};')
    lines.append('inline constexpr std::array<double,3> target_rate_rad_s = {'+','.join(repr(v) for v in target_rate)+'};')
    for key in positive:
        ctype = 'std::uint64_t' if type(data[key]) is int else 'double'
        lines.append(f'inline constexpr {ctype} {key} = {data[key]};')
    for name, values in [('effective_lower', effective_lo), ('effective_upper', effective_hi)]:
        lines.append('inline constexpr std::array<float,12> '+name+' = {'+','.join(repr(v)+'f' for v in values)+'};')
    for name,values in [('tracking_error_max_rad',tracking_rad),('tracking_error_proxy_nm',tracking_proxy)]:
        lines.append('inline constexpr std::array<double,16> '+name+' = {'+','.join(repr(v) for v in values)+'};')
    lines += ['inline constexpr float lower(int i) { return effective_lower[(i/4)*3+i%4]; }',
              'inline constexpr float upper(int i) { return effective_upper[(i/4)*3+i%4]; }', '}}']
    Path(output).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ledger', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    generate(a.ledger, a.output)

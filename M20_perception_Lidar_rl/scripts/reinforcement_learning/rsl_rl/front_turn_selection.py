"""Per-direction turn retention and explicit physical improvement targets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from m20_front_turn_benchmark import validate, identity, PROTOCOL


def regressions(rows, reference):
    before = {identity(r):r for r in reference}
    failures = []
    for row in rows:
        old = before[identity(row)]
        reasons = []
        if row['fell']:
            reasons.append('fell')
        if row['yaw_tracking_ratio'] < max(.65, old['yaw_tracking_ratio']-.05):
            reasons.append('yaw tracking')
        for key, factor, margin in (('backward_distance_m', 1., .025),
                                    ('yaw_error_mean', 1., .03),
                                    ('rear_force_peak_n', 1.1, 30.),
                                    ('touchdown_down_speed_peak_m_s', 1., .05),
                                    ('rear_unled_fraction', 1., .02),
                                    ('rear_lift_fraction', 1., .02)):
            if row[key] > old[key]*factor+margin:
                reasons.append(key)
        if row['pitch_peak_deg'] > old['pitch_peak_deg']+2. or row['roll_peak_deg'] > max(5., old['roll_peak_deg']+3.):
            reasons.append('turn attitude')
        if row['forward'] > 0 and row['forward_error_mean'] > old['forward_error_mean']+.03:
            reasons.append('moving-turn forward tracking')
        if row['forward'] == 0 and row['pure_xy_speed_mean'] > old['pure_xy_speed_mean']+.03:
            reasons.append('pure-turn translation')
        if reasons:
            failures.append('turn/'+str(identity(row))+': '+', '.join(reasons))
    return failures


def error(rows, seed):
    before = {identity(r):r for r in seed}
    terms = []
    for row in rows:
        old = before[identity(row)]
        terms.append(.4*row['backward_distance_m']/max(.04, old['backward_distance_m'])
                     +.25*row['yaw_error_mean']/max(.04, old['yaw_error_mean'])
                     +.2*row['rear_force_peak_n']/max(150., old['rear_force_peak_n'])
                     +.15*row['rear_lift_fraction']/max(.01, old['rear_lift_fraction']))
    return sum(terms)/len(terms)


def targets(rows, seed):
    before = {identity(r):r for r in seed}
    failures = []
    for row in rows:
        old = before[identity(row)]
        reasons = []
        if not .75 <= row['yaw_tracking_ratio'] <= 1.25:
            reasons.append('turn-rate target')
        if row['backward_distance_m'] > .08 or row['backward_speed_peak_m_s'] > .12:
            reasons.append('no-backward target')
        if row['forward'] == 0 and row['pure_xy_speed_mean'] > .08:
            reasons.append('pure-turn centre drift target')
        if row['forward'] > 0 and row['forward_error_mean'] > .1:
            reasons.append('moving-turn speed target')
        if abs(row['yaw']) <= .3 and (row['contact_fraction'] < .97 or row['rear_lift_fraction'] > .02):
            reasons.append('small-turn grounded target')
        if abs(row['yaw']) > .4 and row['front_positive_moment_share'] < .55:
            reasons.append('front-led target')
        if row['rear_unled_fraction'] > .02:
            reasons.append('rear-follow target')
        if row['pitch_peak_deg'] > 6. or row['roll_peak_deg'] > 8.:
            reasons.append('bounded attitude target')
        if row['touchdown_down_speed_peak_m_s'] > .25 or row['rear_force_peak_n'] > max(250., .85*old['rear_force_peak_n']):
            reasons.append('touchdown proxy target')
        if row['fell']:
            reasons.append('fell')
        if reasons:
            failures.append('turn/'+str(identity(row))+': '+', '.join(reasons))
    return failures

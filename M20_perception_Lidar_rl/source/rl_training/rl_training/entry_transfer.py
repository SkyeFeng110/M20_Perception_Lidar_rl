"""Privileged entry reward state; never added to the actor observation."""
import math
import torch


def make_state(n, device):
    z = lambda *shape: torch.zeros((n, *shape), device=device)
    return dict(seen_lower=z(2).bool(), frames=z(2).long(), certified=z(2).bool(),
        first=z().long()-1, lifted=z(2).bool(), valid_lift=z(2).bool(), previous_lift=z(2).bool(),
        early=z().bool(), twin=z().bool(), complete=z().bool(), finished=z().bool(), peak=z(2),
        rear_early=z(2).bool(), front_loss=z().bool(), skipped=z(2).bool(),
        clean_complete=z().bool(), sequence=z().bool(), sequence_sum=z(), roll_peak=z(), hold_sum=z(),
        front_cost_used=z(), rear_cost_used=z(), posture_cost_used=z(),
        skip_cost_used=z(), cost_used=z(), credit_sum=z(), eligible=z().bool())


def transfer_step(s, *, x, bottom, force, wheel_vz, root_speed, front_speed,
                  pitch, pitch_rate, lower, rise, tread, active, reset, dt,
                  rear_x, rear_bottom, rear_force, strict=False):
    """Reward a single transfer, confirmed load, then the other transfer.

    Credits are one-off events or increases in bounded per-wheel progress;
    standing, contact chatter and repeated lifting cannot accumulate reward.
    The two front sides use exactly the same rules. Every cost has an entry
    budget, and failure never removes the independent locomotion objectives.
    """
    for key, value in s.items():
        if not torch.is_tensor(value):
            continue
        if key == 'first':
            value[reset] = -1
        elif value.dtype == torch.bool:
            value[reset] = False
        else:
            value[reset] = 0
    margin = torch.minimum(torch.full_like(tread, .025), .10*tread)
    contact = force >= 15.
    below = (x < 0.) & ((bottom-lower[:, None]).abs() <= .03) & contact
    top = ((x >= margin[:, None]) & (x <= (tread-margin)[:, None])
           & ((bottom-(lower+rise)[:, None]).abs() <= .03)
           & contact & (wheel_vz.abs() <= .25))
    # B1-loaded wheels must not be called airborne just because they are
    # above the lower plane. Vertical-riser contact cannot certify B1.
    lift = (bottom > lower[:, None]+.035) & ~top
    s['seen_lower'] |= active[:, None] & below
    eligible = active & s['seen_lower'].all(1)
    approach = eligible & (x.amax(1) >= -1.30) & (x.amin(1) < tread+.10) & ~s['finished']
    near = eligible & (x.amax(1) >= -.45) & (x.amin(1) < tread+.10) & ~s['finished']
    s['eligible'] |= approach
    # Rear wheels roll and carry load on the lower plane while the front
    # enters. Anticipatory rear swing becomes free inside 22 cm of the riser;
    # this constraint never applies to B2 or any later rear swing.
    # Rear support is released per wheel by its physical distance to the first
    # riser.  Front-wheel certificates are historical and must not release a
    # rear leg while it is still far from B1.
    rear_wait = eligible[:, None] & s['eligible'][:, None] & (rear_x < -.22)
    rear_unsafe = rear_wait & (rear_bottom > lower[:, None]+.035)
    rear_event = rear_unsafe & ~s['rear_early']
    s['rear_early'] |= rear_unsafe
    # Both rear wheels must remain loaded while the front transfer is taught.
    rear_grounded = (((rear_bottom-lower[:, None]).abs() < .03).all(1)
                     & (rear_force >= 5.).all(1))
    certified_before = s['certified'].sum(1)
    s['frames'] = torch.where(top & near[:, None], s['frames']+1, 0)
    ready = s['frames'] >= math.ceil(.06/dt)
    onset = lift & ~s['previous_lift'] & near[:, None] & ~s['certified']
    single = onset.sum(1) == 1
    choose = near & (s['first'] < 0) & single & below.any(1)
    s['first'] = torch.where(choose, onset.long().argmax(1), s['first'])
    first = s['first'].clamp_min(0)
    mate = 1-first
    first_ready_before = s['certified'].gather(1, first[:, None]).squeeze(1)
    first_top_now = top.gather(1, first[:, None]).squeeze(1)
    first_exists = s['first'] >= 0
    mate_lift = lift.gather(1, mate[:, None]).squeeze(1)
    # Check before certifying this tick. A mate lifting during the same tick
    # as the first certification has not waited for established support.
    early_now = near & first_exists & ~first_ready_before & mate_lift
    twin_now = near & lift.all(1)
    early_event = early_now & ~s['early']
    twin_event = twin_now & ~s['twin']
    s['early'] |= early_now
    s['twin'] |= twin_now
    first_loss_event = near & first_exists & first_ready_before & ~first_top_now \
        & ~s['certified'].gather(1, mate[:, None]).squeeze(1) & ~s['front_loss']
    s['front_loss'] |= first_loss_event
    sides = torch.arange(2, device=x.device)[None]
    first_side = sides == s['first'][:, None]
    mate_loaded_below = below.gather(1, mate[:, None]).squeeze(1)
    lead_loaded_top = top.gather(1, first[:, None]).squeeze(1) & first_ready_before
    allowed = torch.where(first_side, mate_loaded_below[:, None], lead_loaded_top[:, None])
    allowed &= first_exists[:, None] & near[:, None]
    lift_event = onset & allowed & ~s['lifted'] & rear_grounded[:, None]
    s['valid_lift'] |= lift_event
    s['lifted'] |= onset
    support_event = ready & ~s['certified'] & s['valid_lift'] & allowed
    # The physical certificate is retained for recovery even after an early
    # mate lift; no permanent lockout on later stairs or forward reward.
    s['certified'] |= ready & near[:, None]
    complete = s['certified'].all(1) & near
    clean_complete = (complete & top.all(1) & ~s['complete'] & (s['first'] >= 0)
                      & s['valid_lift'].all(1) & ~s['front_loss'] & ~s['skipped'].any(1)
                      & ~s['early'] & ~s['twin'] & ~s['rear_early'].any(1))
    s['clean_complete'] |= clean_complete
    s['complete'] |= complete
    s['finished'] |= complete | (eligible & (x.amin(1) > tread+.10))
    roll_potential = ((x.mean(1)+1.30)/.85).clamp(0, 1)
    roll_potential *= approach & below.all(1) & (root_speed > .1)
    roll_credit = (roll_potential-s['roll_peak']).clamp_min(0)
    s['roll_peak'] = torch.maximum(s['roll_peak'], roll_potential)
    # Bounded spatial credit teaches how to clear the riser before the sparse
    # support event exists. Only upward/forward, single-leg transfers count.
    vertical = ((bottom-lower[:, None])/(rise[:, None]+.02).clamp_min(.06)).clamp(0, 1)
    horizontal = ((x+.25)/(.25+.5*tread[:, None])).clamp(0, 1)
    potential = .4*vertical+.6*vertical*horizontal
    useful = allowed & lift & ~s['certified'] & ((root_speed >= -.03) & rear_grounded)[:, None]
    potential = potential*useful
    increment = (potential-s['peak']).clamp_min(0)
    s['peak'] = torch.maximum(s['peak'], potential)
    credit = (.3*lift_event.sum(1)+.6*support_event.sum(1)+.4*clean_complete
              +.35*increment.sum(1)+.35*roll_credit)
    mate_support_event = support_event.gather(1, mate[:, None]).squeeze(1)
    # The second wheel earns sequence credit only after the first has been
    # continuously loaded and confirmed before this tick.  Physical
    # certificates are retained for resynchronization, but invalid transfers
    # never receive the ordered-sequence bonus.
    alternation_event = (mate_support_event & first_ready_before & first_top_now
        & s['valid_lift'].all(1) & ~s['front_loss'] & ~s['skipped'].any(1)
        & ~s['early'] & ~s['twin'] & ~s['rear_early'].any(1))
    s['sequence'] |= alternation_event
    s['sequence_sum'] += alternation_event.float()
    rear_hold = near & (certified_before < 2) & (root_speed > .1) & rear_grounded
    hold_increment = (rear_hold.float()*dt).where(~reset, torch.zeros_like(root_speed))
    hold_credit = hold_increment * (s['hold_sum'] < .5)
    s['hold_sum'] = (s['hold_sum'] + hold_credit).clamp_max(.5)
    # A B1 transfer cannot earn a complete landing bonus by flying beyond its
    # far edge. Charge each wheel once; do not impose a slower speed target.
    skip_now = (near[:, None] & ~s['certified'] & (x > (tread-margin)[:, None])
                & (bottom > (lower+rise-.03)[:, None]))
    skip_event = skip_now & ~s['skipped']
    s['skipped'] |= skip_now
    # Body compression while wheels stay near a riser is different from
    # useful route progress. Use path-frame link speeds, not raw wheel spin.
    blocked = (((root_speed-front_speed.amax(1)-.15)/.40).clamp(0, 1)
               * below.all(1) * approach)
    nose_down = ((pitch-.14).clamp_min(0)/.20).clamp(0, 1)
    nose_rate = ((pitch_rate-.8).clamp_min(0)/1.2).clamp(0, 1)
    rocking = (.7*nose_down+.3*nose_rate)*approach
    # Independent finite ledgers prevent an early twin event from consuming
    # the later rear-timing and posture signal.  Each ledger is bounded so a
    # failure cannot turn the policy into a standstill strategy.
    if strict:
        front_wanted = 1.0*early_event + 1.0*twin_event + 1.0*first_loss_event
        rear_wanted = .9*rear_event.sum(1) + dt*1.0*rear_unsafe.any(1)
        posture_wanted = dt*(.8*blocked+.5*rocking)
    else:
        front_wanted = .45*early_event+.45*twin_event+.45*first_loss_event
        rear_wanted = .25*rear_event.sum(1) + dt*.6*rear_unsafe.any(1)
        posture_wanted = dt*(.6*blocked+.4*rocking)
    front_cost = torch.minimum(front_wanted, (1.5-s['front_cost_used']).clamp_min(0))
    rear_cost = torch.minimum(rear_wanted, (1.0-s['rear_cost_used']).clamp_min(0))
    posture_cost = torch.minimum(posture_wanted, (0.75-s['posture_cost_used']).clamp_min(0))
    skip_cost = torch.minimum(.5*skip_event.sum(1), (1.-s['skip_cost_used']).clamp_min(0))
    s['front_cost_used'] += front_cost
    s['rear_cost_used'] += rear_cost
    s['posture_cost_used'] += posture_cost
    s['skip_cost_used'] += skip_cost
    cost = front_cost + rear_cost + posture_cost + skip_cost
    s['cost_used'] += cost
    s['previous_lift'] = lift.clone()
    # Rear support credit is progress, not an alternating-sequence success.
    credit = credit + .15*hold_credit
    s['credit_sum'] += credit
    return dict(credit=credit, sequence_credit=alternation_event.float(),
        rear_hold=rear_hold, rear_wait=rear_wait.any(1), alternation_event=alternation_event, cost=cost, near=near, lower_support=below,
        top_support=top, air=lift, blocked=blocked, rocking=rocking,
        clean_complete=clean_complete, early_event=early_event, twin_event=twin_event,
        front_loss_event=first_loss_event, skip_event=skip_event.any(1),
        rear_early_event=rear_event.any(1))

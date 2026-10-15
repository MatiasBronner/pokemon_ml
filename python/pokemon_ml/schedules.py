"""Checkpoint-stable bounded exponential learning-rate schedules."""
import math


def learning_rate(schedule, update):
    a, b = schedule['start_lr'], schedule['end_lr']
    start, end = schedule['start_update'], schedule['end_update']
    if not all(math.isfinite(x) and x > 0 for x in (a, b)) or b > a or start < 0 or end <= start:
        raise ValueError('invalid bounded LR schedule')
    progress = min(1., max(0., (update - start) / (end - start)))
    return a * (b / a) ** progress

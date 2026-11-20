"""Startup configuration guard for Dynamic Object Query (DOQ) loss supervision."""

import sys
import time

_DOQ_DISABLED_LOSS_WARNING = (
    "DOQ mismatch [warning {idx}/3]: use_dynamic_query_selection=True with "
    "doq_loss_coef == 0.0; loss_doq is disabled and unused doq_out autograd "
    "graphs will be detached and discarded."
)


_DEFAULT_SLEEP = time.sleep


def warn_if_doq_loss_disabled(args, logger=None, sleep_fn=time.sleep):
    """Warn three times and sleep 3s when DOQ is enabled with zero loss weight."""
    use_doq = bool(getattr(args, "use_dynamic_query_selection", False))
    raw_coef = getattr(args, "doq_loss_coef", 0.0)
    doq_loss_coef = 0.0 if raw_coef is None else float(raw_coef)

    if not (use_doq and doq_loss_coef == 0.0):
        return False

    for idx in range(1, 4):
        message = _DOQ_DISABLED_LOSS_WARNING.format(idx=idx)
        if logger is not None:
            logger.warning(message)
        else:
            print(message, file=sys.stderr, flush=True)

    if sleep_fn is _DEFAULT_SLEEP:
        time.sleep(3)
    else:
        sleep_fn(3)
    return True

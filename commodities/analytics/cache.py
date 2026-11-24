"""Revision-keyed memoization — the app's entire caching model.

Analytics functions take the Store as their first argument. The cache key is
(function, db path, remaining args, store revision): when a refresh writes data
the revision bumps, old keys never match again, and recomputation happens
exactly when data actually changed. No TTLs, no token files, no manual
invalidation.

Two pools, deliberately — do not merge them back into one.

`_cache` holds RESULTS: features, options_timeseries, screen, inventory_*,
seasonality, firm_hhi. There are only a handful per asset, they are what a page
render waits on, and they are the expensive things to lose.

`_scratch` holds INTERMEDIATES — today just enriched_chain, which is keyed per
(asset, expiry month, trade date). One options_timeseries() sweep of a single
asset produces ~55 of them; the whole database holds 263 today and gains about
20 a day as new trade dates are stored.

Sharing one budget meant the intermediates evicted the results. Measured on the
live database with the old single 64-entry pool: rotating through the five
assets and then repeating the rotation was just as slow the second time
(gold 49.0s, having cost 51.5s moments earlier), because computing any one
asset flushed every other asset's finished result. Raising a single shared
limit would only postpone that — the chain count grows daily and would re-cross
any ceiling within weeks. Separate pools make it structural: chains can only
ever evict other chains, at any database size.
"""

from collections import OrderedDict
from functools import wraps

# Results are few and small (a features frame is ~500 rows).
_MAX_ENTRIES = 128

# Intermediates are large: enriched_chain frames measure ~295 KB on average and
# 339 KB at the largest (gold's 856-strike December chain). 384 slots cap this
# pool near 113 MB and hold every chain currently in the database, with room to
# grow. Raise it if the working set outgrows that and the memory is available.
_SCRATCH_MAX_ENTRIES = 384

_cache = OrderedDict()
_scratch = OrderedDict()


def cached(fn=None, *, scratch=False):
    """Memoize on (function, db path, args, store revision).

    scratch=True routes a function's entries to the intermediates pool, where
    they can never evict an analytics result. Use it for anything that produces
    one entry per chain/date/slice rather than one entry per asset.
    """
    def decorate(f):
        pool = _scratch if scratch else _cache
        limit = _SCRATCH_MAX_ENTRIES if scratch else _MAX_ENTRIES

        @wraps(f)
        def wrapper(store, *args, **kwargs):
            key = (f.__qualname__, store.db_path, args,
                   tuple(sorted(kwargs.items())), store.revision())
            if key in pool:
                pool.move_to_end(key)
                return pool[key]
            value = f(store, *args, **kwargs)
            pool[key] = value
            if len(pool) > limit:
                pool.popitem(last=False)
            return value
        return wrapper

    return decorate(fn) if fn is not None else decorate


def clear():
    _cache.clear()
    _scratch.clear()

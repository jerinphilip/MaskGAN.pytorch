from collections import defaultdict
from gc import get_objects
import functools
from pprint import pprint


def leak_check(f):
    @functools.wraps(f)
    def __inner(*args, **kwargs):
        print("Running Leak Check", f.__name__)
        before, after = defaultdict(int), defaultdict(int)
        for _object in get_objects():
            before[type(_object)] += 1

        result = f(*args, **kwargs)

        for _object in get_objects():
            after[type(_object)] += 1
        
        delta = defaultdict(int)
        for key in after:
            _delta = after[key] - before[key]
            if _delta > 0:
                delta[key] = (_delta, after[key])

        pprint(delta)
        print("Finished Leack Check!")
        return result
    return __inner


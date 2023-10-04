from wrapt_timeout_decorator import *


class TimeoutError(BaseException):
    pass

def simplify_with_timeout(f, sec):
    @timeout(sec)
    def _simplify(f):
        try:
            sf = sp.simplify(f)
            return sf
        except TimeoutError:
            return f
        except Exception as e:
            return f
    return _simplify(f)

def solve_with_timeout(f, var, sec):
    @timeout(sec)
    def _solve(f, var):
        try:
            sol = sp.solve(f, var, check=False, simplify=False)
            return sol
        except TimeoutError:
            return None
        except Exception as e:
            return None
    return _solve(f, var)

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.calculus.util import AccumBounds
import time
import signal
from copy import copy

#from with_timeout import simplify_with_timeout, solve_with_timeout
from wrapt_timeout_decorator import *

BINARY_OPERATORS = {
    'add': '({})+({})',
    'sub': '({})-({})',
    'mul': '({})*({})',
    'div': '({})/({})',
    'pow': '({})**({})'
    }

UNARY_OPERATORS = dict(sqrt='sqrt({})', exp='exp({})', ln='ln({})', abs='Abs({})',
                       sign='sign({})', sin='sin({})', cos='cos({})', tan='tan({})',
                       cot='cot({})', sec='sec({})', csc='csc({})', asin='asin({})',
                       acos='acos({})', atan='atan({})', acot='acot({})', asec='asec({})',
                       acsc='acsc({})', sinh='sinh({})', cosh='cosh({})', tanh='tanh({})',
                       coth='coth({})', sech='sech({})', csch='csch({})', asinh='asinh({})',
                       acosh='acosh({})', atanh='atanh({})', acoth='acoth({})', asech='asech({})',
                       acsch='acsch({})', pow2='({})**2', pow3='({})**3', pow4='({})**4')

SYMPY_OPERATORS = {
    sp.Add: 'add',
    sp.Mul: 'mul',
    sp.Pow: 'pow',
    sp.exp: 'exp',
    sp.log: 'ln',
    sp.Abs: 'abs',
    sp.sign: 'sign',
    sp.sin: 'sin',
    sp.cos: 'cos',
    sp.tan: 'tan',
    sp.cot: 'cot',
    sp.sec: 'sec',
    sp.csc: 'csc',
    sp.asin: 'asin',
    sp.acos: 'acos',
    sp.atan: 'atan',
    sp.acot: 'acot',
    sp.asec: 'asec',
    sp.acsc: 'acsc',
    sp.sinh: 'sinh',
    sp.cosh: 'cosh',
    sp.tanh: 'tanh',
    sp.coth: 'coth',
    sp.sech: 'sech',
    sp.csch: 'csch',
    sp.asinh: 'asinh',
    sp.acosh: 'acosh',
    sp.atanh: 'atanh',
    sp.acoth: 'acoth',
    sp.asech: 'asech',
    sp.acsch: 'acsch',
    sp.Derivative: 'der'
    }

OTHERS = {
    'f': 'f({})',
    'der': 'Derivative({},{})'
}

BINARY_OPERATORS_PROBS = {
    'add': 700,
    'sub': 600,
    'mul': 1100,
    'div': 160,
    'pow': 50,
    }

UNARY_OPERATORS_PROBS = dict(sqrt=100, exp=200, ln=200,
                       sin=80, cos=80, tan=80,
                       asin=20, acos=20, atan=20,
                       sinh=10, cosh=10, tanh=10,
                       asinh=5, acosh=5, atanh=5,
                       pow2=500, pow3=90, pow4=30)

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

def solve_with_timeout(f, var, sec=1):
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


def decompose_int(val):
    int_str = str(val)
    if int_str[0] == '-':
        res = ['-'] + [*int_str[1:]]
    else:
        res = ['+'] + [*int_str]
    return res


def get_int(rep):
    ln = 0
    for x in rep[1:]:
        if not x.isdigit():
            break
        ln += 1
    if rep[0] == '-':
        val = '-' + ''.join(rep[1:ln + 1])
    else:
        val = ''.join(rep[1:ln + 1])
    return val, ln + 1


def get_unbinary_dist(n_ops):
    D = [[0] * (2 * n_ops + 1)]
    for _ in range(2 * n_ops):
        D.append([1])
    for n in range(1, 2 * n_ops):
        for e in range(1, 2 * n_ops - n + 1):
            D[e].append(D[e - 1][n] + D[e][n - 1] + D[e + 1][n - 1])
    return D


class ExpressionGenerator(object):
    def __init__(self, n_ops=10, max_int=5, max_seq_len=256, unary_operators=None, binary_operators=None):
        self.n_ops = n_ops
        self.max_seq_len = max_seq_len
        self.max_int = max_int
        self.un_ops = unary_operators if unary_operators else list(UNARY_OPERATORS.keys())
        self.bin_ops = binary_operators if binary_operators else list(BINARY_OPERATORS.keys())
        self.un_probs = np.array([UNARY_OPERATORS_PROBS[op] if op in UNARY_OPERATORS_PROBS else 0 for op in self.un_ops])
        self.un_probs = self.un_probs / self.un_probs.sum()
        self.bin_probs = np.array([BINARY_OPERATORS_PROBS[op] if op in BINARY_OPERATORS_PROBS else 0 for op in self.bin_ops])
        self.bin_probs = self.bin_probs / self.bin_probs.sum()

        self.spec_symbols = ['+', '-', "y'", "y''", 'end_x_start_y']
        self.all_operators = list(BINARY_OPERATORS.keys()) + list(UNARY_OPERATORS.keys())

        self.vars = {
            'x': sp.Symbol('x', real=True, nonzero=True),
            'y': sp.Symbol('y', real=True, nonzero=True),
            'z': sp.Symbol('z', real=True, nonzero=True),
            't': sp.Symbol('t', real=True, nonzero=True)
        }
        self.coef = {
            'C1': sp.Symbol('C1', real=True),
            'C2': sp.Symbol('C2', real=True)
        }
        self.functions = {
            'f': sp.Function('f', real=True, nonzero=True)
        }
        self.constants = ['pi', 'E']
        self.unbinary_tree_dist = get_unbinary_dist(n_ops)

        self.leaf_type_probs = [0.6, 0.4]

        self.local_dict = {}
        for sp_repr in list(self.vars.values()) + list(self.coef.values()) + list(self.functions.values()):
            self.local_dict[sp_repr.name] = sp_repr

        self.ints = [str(i) for i in range(10)]

        self.tokens = ['<s>', '</s>', '<pad>'] + self.spec_symbols + list(BINARY_OPERATORS.keys()) +\
                      list(UNARY_OPERATORS.keys()) + list(self.coef.keys()) + list(self.vars.keys()) + self.constants + self.ints

        self.id2token = {i: s for i, s in enumerate(self.tokens)}
        self.token2id = {s: i for i, s in self.id2token.items()}

    def internal_node_to_allocate(self, e, n, rand_gen):
        un_probs = np.array([self.unbinary_tree_dist[e - k][n - 1] for k in range(e)])
        un_probs = un_probs / self.unbinary_tree_dist[e][n]
        bin_probs = np.array([self.unbinary_tree_dist[e - k + 1][n - 1] for k in range(e)])
        bin_probs = bin_probs / self.unbinary_tree_dist[e][n]
        all_probs = np.hstack((un_probs, bin_probs))
        k = rand_gen.choice(2 * e, p=all_probs)
        is_unary = k < e
        k = k % e
        return k, is_unary

    def get_tree(self, n_ops, rand_gen):
        e = 1
        tree = ['un']
        skipped = 0
        inds = ['un', 'bin_1', 'bin_2']
        for n in range(n_ops, 0, -1):
            k, is_unary = self.internal_node_to_allocate(e, n, rand_gen)
            if is_unary:
                op = rand_gen.choice(self.un_ops, p=self.un_probs)
                e = e - k
                a = 1
            else:
                op = rand_gen.choice(self.bin_ops, p=self.bin_probs)
                e = e - k + 1
                a = 2
            skipped += k
            pos = [i for i, v in enumerate(tree) if v in inds][skipped]


            if tree[pos] == 'bin_1':
                if pos-1 >= 0 and tree[pos-1] == 'bin_1':
                    tree[pos-1] = 'bin_2'
                elif pos+1 < len(tree) and tree[pos+1] == 'bin_1':
                    tree[pos+1] = 'bin_2'

            ind = 'un' if a == 1 else 'bin_1'

            tree = tree[:pos] + [op] + [ind for _ in range(a)] + tree[pos + 1:]

            if pos >= 1 and tree[pos] == tree[pos-1]:
                if tree[pos] in self.bin_ops:
                    if tree[pos] == 'div':
                        rep_ops = rand_gen.choice(self.bin_ops, size=2, replace=False, p=self.bin_probs)
                        tree[pos] = rep_ops[0] if rep_ops[0] != tree[pos] else rep_ops[1]
                else:
                    rep_ops = rand_gen.choice(self.un_ops, size=2, replace=False, p=self.un_probs)
                    tree[pos] = rep_ops[0] if rep_ops[0] != tree[pos] else rep_ops[1]

        return tree


    def decorate_tree(self, tree, rand_gen, req_y=False, req_c=False, solvable_in_y=True):

        if req_y and not solvable_in_y:
            vars = ['x', 'y']
        else:
            vars = ['x']

        var_ids = []
        c_ids = []

        if req_c:
            for i in range(len(tree)):
                if tree[i] in ['bin_1', 'bin_2']:
                    c_ids.append(i)
            c_pos = rand_gen.choice(c_ids)
            if tree[c_pos] == 'bin_1':
                if tree[c_pos-1] == 'bin_1':
                    tree[c_pos-1] = 'C1'
                else:
                    tree[c_pos] = 'C1'
            else:
                tree[c_pos] = 'C1'

        i = -1

        while i < len(tree) - 1:
            i += 1
            if tree[i] == 'un':
                tree[i] = rand_gen.choice(vars)
                var_ids.append(i)
            elif tree[i] == 'bin_2':
                tree[i], is_var = self.get_leaf(vars, rand_gen)
                if is_var:
                    var_ids.append(i)
            elif tree[i] == 'bin_1':
                if tree[i-1] == 'C1':
                    tree[i] = rand_gen.choice(vars)
                    var_ids.append(i)
                else:
                    arg1 = rand_gen.choice(vars)
                    arg2, is_var = self.get_leaf(vars, rand_gen)
                    if is_var:
                        tree[i], tree[i+1] = arg1, arg2
                        var_ids.extend([i, i+1])
                    else:
                        j = rand_gen.randint(2)
                        tree[i + j] = arg1
                        tree[i + 1 - j] = arg2
                        var_ids.append(i+j)
                    i += 1

        if req_y and solvable_in_y:
            y_id = rand_gen.choice(var_ids)
            tree[y_id] = 'y'

        for i in range(len(tree) - 1, -1, -1):
            if isinstance(tree[i], int):
                tree = tree[:i] + decompose_int(tree[i]) + tree[i + 1:]

        return tree

    def get_leaf(self, vars, rand_gen):
        leaf_type = rand_gen.choice(2, p=self.leaf_type_probs)
        if leaf_type == 0:
            c = rand_gen.randint(1, self.max_int + 1)
            c = rand_gen.choice([-c, c])
            return (int(c), False)
        elif leaf_type == 1:
            return (rand_gen.choice(vars), True)

    def get_prefix_expr(self, n_ops, rand_gen, req_y=False, req_c=False, solvable_in_y=True):
        tree = self.get_tree(n_ops, rand_gen)
        return self.decorate_tree(tree, rand_gen, req_y, req_c, solvable_in_y)


    def prefix_to_infix(self, exp):
        v = exp[0]
        v_type = None
        if v in UNARY_OPERATORS.keys():
            v_type, a = 1, 1
        elif v in BINARY_OPERATORS.keys():
            v_type, a = 2, 2
        elif v == 'f':
            v_type, a = 3, 1
        elif v == 'der':
            v_type, a = 3, 2

        if v_type is not None:
            args = []
            nxt = exp[1:]
            for _ in range(a):
                arg, nxt = self.prefix_to_infix(nxt)
                args.append(arg)
            if v_type == 1:
                to_str = UNARY_OPERATORS[v].format(*args)
            elif v_type == 2:
                to_str = BINARY_OPERATORS[v].format(*args)
            else:
                to_str = OTHERS[v].format(*args)
            return to_str, nxt
        elif v in self.vars or v in self.coef or v in self.constants:
            return v, exp[1:]
        else:
            int_val, end = get_int(exp)
            return int_val, exp[end:]

    def has_nan(self, expr):
        if expr.has(sp.I) or expr.has(sp.nan) or expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.zoo):
            return True


    def infix_to_sympy(self, infix):
        expr = parse_expr(infix, local_dict=self.local_dict, evaluate=True)
        if self.has_nan(expr) or expr.has(AccumBounds):
            raise Exception('Invalid expression')
        return expr

    def sympy_to_prefix(self, expr):
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return decompose_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + decompose_int(int(expr.p)) + decompose_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']

        for op_type, op_name in SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                n_args = len(expr.args)
                if op_name == 'der':
                    parse_list = self.sympy_to_prefix(expr.args[0])
                    for var, degree in expr.args[1:]:
                        parse_list = ['der' for _ in range(int(degree))] + parse_list + [str(var) for _ in
                                                                                         range(int(degree))]
                    return parse_list

                parse_list = []
                for i in range(n_args):
                    if i == 0 or i < n_args - 1:
                        parse_list.append(op_name)
                    parse_list += self.sympy_to_prefix(expr.args[i])
                return parse_list

        if isinstance(expr, self.functions['f']):
            n_args = len(expr.args)
            parse_list = []
            for i in range(n_args):
                if i == 0 or i < n_args - 1:
                    parse_list.append('f')
                parse_list += self.sympy_to_prefix(expr.args[i])
            return parse_list

        raise Exception(f"No such sympy operator: {expr}")

    def replace_f_by_y(self, prefix_expr):
        pos = 0
        cur_len = len(prefix_expr)
        while pos < cur_len:
            if prefix_expr[pos] == 'der':
                if prefix_expr[pos + 1] == 'der':
                    prefix_expr = prefix_expr[:pos] + ["y''"] + prefix_expr[pos + 6:]
                    cur_len -= 5
                else:
                    prefix_expr = prefix_expr[:pos] + ["y'"] + prefix_expr[pos + 4:]
                    cur_len -= 3
            elif prefix_expr[pos] == 'f':
                prefix_expr = prefix_expr[:pos] + ["y"] + prefix_expr[pos + 2:]
                cur_len -= 1
            pos += 1
        return prefix_expr

    def replace_y_by_f(self, prefix_expr):
        pos = 0
        cur_len = len(prefix_expr)
        while pos < cur_len:
            if prefix_expr[pos] == "y''":
                prefix_expr = prefix_expr[:pos] + ['der', 'der', 'f', 'x', 'x', 'x'] + prefix_expr[pos + 1:]
                cur_len += 5
                pos += 6
            elif prefix_expr[pos] == "y'":
                prefix_expr = prefix_expr[:pos] + ['der', 'f', 'x', 'x'] + prefix_expr[pos + 1:]
                cur_len += 3
                pos += 4
            elif prefix_expr[pos] == 'y':
                prefix_expr = prefix_expr[:pos] + ['f', 'x'] + prefix_expr[pos + 1:]
                cur_len += 1
                pos += 2
            else:
                pos += 1
        return prefix_expr

    def prefix_to_simpy(self, prefix, has_y=False):
        eq = copy(prefix)
        if has_y:
            eq = self.replace_y_by_f(eq)
        infix, _ = self.prefix_to_infix(eq)
        expr = self.infix_to_sympy(infix)
        return expr

    def condense(self, eq, *x):
        reps = {}
        con = sp.numbered_symbols('c')
        free = eq.free_symbols
        c1 = self.coef['C1']
        def c():
            while True:
                rv = next(con)
                if rv not in free:
                    return rv
        def do(e):
            i, d = e.as_independent(*x)
            if not i.args: return e
            return e.func(reps.get(i, reps.setdefault(i, c())), d)
        rv = eq.replace(lambda x: x.is_Add or x.is_Mul, lambda x: do(x))
        reps = {v: k for k, v in reps.items()}
        keep = rv.free_symbols & set(reps)
        reps = {k: reps[k].xreplace(reps) for k in keep}
        if len(reps) == 1:
            con, rep = list(reps.items())[0]
            return rv.subs(con, c1), rep
        else:
            return eq, None

    def get_separable_eq(self, n_ops, rand_gen):
        x = self.vars['x']
        z = self.vars['z']
        c1 = self.coef['C1']

        p_ops = rand_gen.randint(1, n_ops)
        q_ops = n_ops - p_ops

        p = self.get_prefix_expr(p_ops, rand_gen)
        p = self.prefix_to_simpy(p)

        q = self.get_prefix_expr(q_ops, rand_gen)
        q = self.prefix_to_simpy(q)

        if x not in p.free_symbols:
            return None

        p = p.subs(x, z)
        A = p.diff(z)
        B = q.diff(x)

        if x not in q.free_symbols:
            sol = p + c1
        else:
            sol = p - q + c1

        eq = B / A

        sol, _ = self.condense(sol, x, z)
        return eq, sol

    def get_lde_1(self, n_ops, rand_gen):
        x = self.vars['x']
        z = self.vars['z']
        c1 = self.coef['C1']

        p_ops = rand_gen.randint(1, n_ops)
        q_ops = n_ops - p_ops

        p = self.get_prefix_expr(p_ops, rand_gen)
        p = self.prefix_to_simpy(p)

        q = self.get_prefix_expr(q_ops, rand_gen)
        q = self.prefix_to_simpy(q)

        is_q_const = x not in q.free_symbols
        is_p_const = x not in p.free_symbols

        p_i = p.diff(x)
        q_i = q.diff(x)

        A = q_i / q
        B = q * p_i

        if is_q_const and is_p_const:
            sol = z + c1
        elif is_q_const:
            sol = z + c1 - q * p
        elif is_p_const:
            sol = z + c1 * q
        else:
            sol =  z + q * c1 - q * p

        eq = A * z + B
        sol, _ = self.condense(sol, x, z)

        return eq, sol

    def get_ode1(self, n_ops, rand_gen, sol_solvable_in_y=False):

        x = self.vars['x']
        y = self.vars['y']
        z = self.vars['z']
        f = self.functions['f']
        z_new = z

        while y not in z_new.free_symbols:
            try:
                z_new_prefix = self.get_prefix_expr(n_ops, rand_gen, req_y=True, req_c=False, solvable_in_y=sol_solvable_in_y)
                z_new = self.prefix_to_simpy(z_new_prefix)
            except:
                pass

        z_new = z_new.subs(y, f(x))

        basis_eq_type = rand_gen.randint(2)

        if basis_eq_type == 0:
            gen = lambda x: self.get_separable_eq(x, rand_gen)
        else:
            gen = lambda x: self.get_lde_1(x, rand_gen)

        res = None
        while res is None:
            try:
                res = gen(n_ops)
            except:
                pass

        dz_orig, z_orig = res
        dz_new = z_new.diff(x)
        eq = dz_orig.subs(z, z_new) - dz_new
        eq = sp.fraction(eq)[0]
        eq = parse_expr(str(eq).replace('Abs', ''), local_dict=self.local_dict, evaluate=True)
        sol = z_orig.subs(z, z_new)
        sol, _ = self.condense(sol, x, y)

        return eq, sol

    def get_solved_in_y(self, n_ops, rand_gen):
        x = self.vars['x']
        y = self.vars['y']
        t = self.vars['t']
        c1 = self.coef['C1']

        int_x = self.get_prefix_expr(n_ops, rand_gen, req_c=True)
        int_x = self.prefix_to_simpy(int_x).subs(x, t)

        if c1 not in int_x.free_symbols:
            return None
        int_x, _ = self.condense(int_x, t)

        x_t = int_x.diff(t)
        x_t = simplify_with_timeout(x_t, sec=1)
        y_t = t*x_t - int_x
        y_t = simplify_with_timeout(y_t, sec=1)

        if t not in x_t.free_symbols:
            return None

        if c1 not in x_t.free_symbols:
            eq = x - x_t
            y_t, _ = self.condense(y_t, t)

        elif c1 not in y_t.free_symbols:
            eq = y - y_t
            x_t, _ = self.condense(x_t, t)

        else:
            x_cond, x_rep = self.condense(x_t, t)
            y_cond, y_rep = self.condense(y_t, t)
            if x_rep and y_rep and x_rep.equals(y_rep):
                x_t = x_cond
                y_t = y_cond
            solve_c1 = solve_with_timeout(y - t*x + int_x, c1)
            if solve_c1 is None:
                return None

            if len(solve_c1) == 0 or type(solve_c1) is not list:
                return None
            solve_c1 = [s for s in solve_c1 if t in s.free_symbols]
            if len(solve_c1) == 0:
                return None
            c_xt = solve_c1[rand_gen.randint(len(solve_c1))]
            if type(c_xt) is tuple or type(c_xt) is sp.Piecewise:
                return None

            eq = x - x_t
            eq = eq.subs(c1, c_xt)

        eq = simplify_with_timeout(eq, sec=1)
        eq = sp.fraction(eq)[0]
        #eq = parse_expr(str(eq).replace('Abs', ''), local_dict=self.local_dict, evaluate=True)
        if t not in eq.free_symbols:
            return None

        return eq, x_t, y_t

    def get_solved_in_x(self, n_ops, rand_gen):
        x = self.vars['x']
        y = self.vars['y']
        t = self.vars['t']
        c1 = self.coef['C1']

        int_y = self.get_prefix_expr(n_ops, rand_gen, req_c=True)
        int_y = self.prefix_to_simpy(int_y).subs(x, t)
        if c1 not in int_y.free_symbols:
            return None
        int_y, _ = self.condense(int_y, t)

        int_y_i = int_y.diff(t)
        y_t = int_y_i * t**2
        y_t = simplify_with_timeout(y_t, sec=1)
        x_t = t * int_y_i + int_y
        x_t = simplify_with_timeout(x_t, sec=1)

        if t not in y_t.free_symbols:
            return None

        if c1 not in x_t.free_symbols:
            eq = x - x_t
            y_t, _ = self.condense(y_t, t)

        elif c1 not in y_t.free_symbols:
            eq = y - y_t
            x_t, _ = self.condense(x_t, t)

        else:
            x_cond, x_rep = self.condense(x_t, t)
            y_cond, y_rep = self.condense(y_t, t)
            if x_rep and y_rep and x_rep.equals(y_rep):
                x_t = x_cond
                y_t = y_cond

            solve_c1 = solve_with_timeout(x - y/t - int_y, c1)
            if solve_c1 is None:
                return None

            if len(solve_c1) == 0 or type(solve_c1) is not list:
                return None
            solve_c1 = [s for s in solve_c1 if t in s.free_symbols]
            if len(solve_c1) == 0:
                return None
            c_xt = solve_c1[rand_gen.randint(len(solve_c1))]
            if type(c_xt) is tuple or type(c_xt) is sp.Piecewise:
                return None

            eq = y - y_t
            eq = eq.subs(c1, c_xt)


        eq = simplify_with_timeout(eq, sec=1)
        eq = sp.fraction(eq)[0]
        eq = parse_expr(str(eq).replace('Abs', ''), local_dict=self.local_dict, evaluate=True)

        if t not in eq.free_symbols:
            return None

        return eq, x_t, y_t


    def get_unresolved_derivative(self, n_ops, rand_gen):
        eq_type = rand_gen.randint(2)
        if eq_type == 0:
            try:
                res = self.get_solved_in_y(n_ops, rand_gen)
            except:
                return None
        else:
            try:
                res = self.get_solved_in_x(n_ops, rand_gen)
            except:
                return None

        return res


    def get_prefix_ode_1(self, n_ops, rand_gen, sol_solvable_in_y=False):
        try:
            res = self.get_ode1(n_ops, rand_gen)
        except:
            return None

        if res is None:
            return None
        else:
            eq, sol = res

        if self.has_nan(eq) or self.has_nan(sol):
            return None
        prefix_sol = self.sympy_to_prefix(sol)
        prefix_sol = self.replace_f_by_y(prefix_sol)
        prefix_eq = self.sympy_to_prefix(eq)
        prefix_eq = self.replace_f_by_y(prefix_eq)

        if len(prefix_sol) > self.max_seq_len - 2:
            return None
        if len(prefix_eq) > self.max_seq_len - 2:
            return None
        return prefix_eq, prefix_sol


    def get_prefix_unresolved_derivative(self, n_ops, rand_gen):

        try:
            res = self.get_unresolved_derivative(n_ops, rand_gen)
        except:
            return None

        if res is None:
            return None
        else:
            eq, x_t, y_t = res

        if self.has_nan(eq) or self.has_nan(x_t + y_t):
            return None

        prefix_x = self.sympy_to_prefix(x_t)
        prefix_y = self.sympy_to_prefix(y_t)
        prefix_eq = self.sympy_to_prefix(eq)

        if len(prefix_eq) > self.max_seq_len - 2:
            return None
        if len(prefix_x) + len(prefix_y) + 1 > self.max_seq_len - 2:
            return None
        prefix_sol = prefix_x + ['end_x_start_y'] + prefix_y
        return prefix_eq, prefix_sol

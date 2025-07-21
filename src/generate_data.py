from expressions_generator import ExpressionGenerator
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sympy.core.cache import clear_cache


BIN_OPERATORS = ['add', 'sub', 'mul', 'div']

UN_OPERATORS = ['sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']


def create_eq_string(n_ops, rand_gen):
    try:
        res = gen.get_prefix_unresolved_derivative(n_ops, rand_gen)
    except:
        return None
    if res is not None:
        prefix_eq, prefix_sol = res
        prefix_eq_id = [sos_id] + [token2id[tok] for tok in prefix_eq] + [eos_id]
        prefix_sol_id = [sos_id] + [token2id[tok] for tok in prefix_sol] + [eos_id]
        eq_str = ' '.join(prefix_eq_id) + ',' + ' '.join(prefix_sol_id)
        return eq_str


def write_data(num_eqs, file_name, rand_gen):
    count = 0
    with open(file_name, 'a') as file:
        for i in tqdm(range(num_eqs)):
            n_ops = rand_gen.randint(2, 6)
            res = create_eq_string(n_ops, rand_gen)
            if res is not None:
                count += 1
                file.write(res)
                file.write('\n')
            if i % 20 == 0:
                clear_cache()
    return count

def add_count(count):
    counts.append(count)

def get_params(n, eq_to_generate, random_seed):
    params = []
    for i in range(n):
        file_name = 'data_p_{}'.format(i+1)
        rand_gen = np.random.RandomState(random_seed+i+1)
        param = (eq_to_generate, file_name, rand_gen)
        params.append(param)
    return params


if __name__ == "__main__":

    gen = ExpressionGenerator(n_ops=13, unary_operators=UN_OPERATORS, binary_operators=BIN_OPERATORS)
    token2id = {key: str(value) for key, value in gen.token2id.items()}
    sos_id = str(gen.token2id['<s>'])
    eos_id = str(gen.token2id['</s>'])
    params = get_params(4, 100000, 19292909)
    counts = []

    pool = mp.Pool(4)
    for arg in params:
        pool.apply_async(write_data, args=arg, callback=add_count)
    pool.close()
    pool.join()
    print('Total number of generated eqs: ', sum(counts))


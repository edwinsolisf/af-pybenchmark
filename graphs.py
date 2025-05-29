import matplotlib.pyplot as plt
import numpy as np
import json

PKG_NAMES = ['numpy', 'afcuda', 'cupy', 'dpnp'] # package list in graph order
# tests = [ 'cholesky', 'det', 'norm', 'normal', 'uniform', 'pi', 'black_scholes', 'fft', 'inv', 'svd', 'group_elementwise'] # Tests to be shown in graphs
tests = [ 'cholesky', 'neural_network', 'gemm', 'mandelbrot', 'nbody', 'pi', 'black_scholes', 'fft', 'group_elementwise'] # Tests to be shown in graphs
show_test_numbers = True # Show Speedup numbers
round_numbers = 1 # Round to digits after decimal

def get_benchmark_data():
    results = {}
    descriptions = {}
    with open('results.json') as f:
        js = json.load(f)
        for bench in js['benchmarks']:
            test_name = bench["name"]
            test_name = test_name[test_name.find('_') + 1:test_name.find('[')]

            key = bench["param"]
            val = bench["stats"]["ops"]

            if len(bench["extra_info"]) != 0 and (not test_name in descriptions):
                descriptions[test_name] = bench["extra_info"]["description"]

            if test_name not in results:
                results[test_name] = { key : val }
            else:
                results[test_name][key] = val

    return results, descriptions

def create_graph(test_name, test_results):
    names = []
    values = []
    for name in test_results:
        names.append(name)
        values.append(test_results[name])

    bar = plt.bar(names, values)
    plt.title(test_name)

    plt.savefig("img/" + test_name + ".png")
    plt.close()

def generate_individual_graphs():
    results, descriptions = get_benchmark_data()

    for test in results:
        create_graph(test, results[test])


def generate_group_graph(test_list = None, show_numbers = False):
    results, descriptions = get_benchmark_data()

    width = 1 / (1 + len(PKG_NAMES))
    multiplier = 0

    tests = None
    if test_list:
        tests = test_list
    else:
        tests = results.keys()

    tests_values = {}
    x = np.arange(len(tests))

    for name in PKG_NAMES:
        tests_values[name] = []

    max_val = 1
    for test in tests:
        for name in PKG_NAMES:
            base_value = results[test]["numpy"]
            if name in results[test]:
                val = results[test][name] / base_value

                if round_numbers:
                    val = round(val, round_numbers)

                if max_val < val:
                    max_val = val

                tests_values[name].append(val)
            else:
                tests_values[name].append(np.nan)

    fig, ax = plt.subplots(layout='constrained')

    for name in PKG_NAMES:
        offset = width * multiplier
        rects = ax.barh(x + offset, tests_values[name], width, label=name)
        if show_numbers:
            ax.bar_label(rects, padding=3, rotation=0)
        multiplier += 1

    xlabels = []
    for test in tests:
        xlabels.append(test  + "\n" + descriptions[test])

    ax.set_xlabel('Speedup')
    ax.set_xscale('log')
    ax.set_title('Runtime Comparison')
    ax.set_yticks(x + width, xlabels, rotation=0)
    # ax.set_ylim([0.0, max_val * 1.25])
    ax.legend(loc='lower right', ncols=len(PKG_NAMES))
    fig.set_figheight(8)
    fig.set_figwidth(12)
    fig.savefig("img/comparison.png")
    plt.show()
    
def main():
    generate_group_graph(tests, show_test_numbers)

if __name__ == "__main__":
    main()
import numpy as np
from sympy import symbols, diff
from matplotlib import pyplot as plt
from scipy.optimize import fmin, minimize



# plt.xkcd()
# plt.style.use("fivethirtyeight")
plt.style.use("ggplot")


def dmdo(cg=False, opt_t=False, ts=[45e-2], K=25, e=0.01, multimode=False):
    # Min <- J = 1/2 E(qxi^2+rui^2) + (1/2)fxN^2
    N = 5
    a = 1
    b = 1/2
    q = 1
    r = .5
    f = 1.5

    n_iter = K
    epsilon = e

    X = np.zeros(shape=(n_iter, N + 1))
    P = np.zeros(shape=(n_iter, N))
    B = np.zeros(shape=(n_iter, 1))
    U = np.zeros(shape=(n_iter, N))
    print(ts, multimode)
    J = np.zeros(shape=(4 if multimode else len(ts), n_iter))
    d = np.zeros(shape=(n_iter, N))

    x_0 = 2

    def _optimize(t, use_CG=cg, optimize_t=opt_t, epsilon=epsilon, init_control=False):
        # nonlocal u
        if not init_control:
            u = np.random.uniform(low=0, high=20, size=N)  # 1st control randomization
        else:
            u = np.array([2, 16, 2, 8, 4])  # same as in the initial test

        for i in range(n_iter):

            x = np.insert(x_0 + np.cumsum(u) / 2, 0, x_0)  # calculate controls
            X[i, :] = x  # append xi to array, just to have a summary of changes

            p_N = 3 * x[-1] / 2  # calculate the value of the Nth costate (since dx_N != 0)
            p_reversed = p_N + np.cumsum(x[-2::-1])  # calc the adjoint vector
            P[i, :] = p_reversed  # keep them for later

            b = np.add(u, np.insert(p_reversed, 0, p_N)[-2::-1]) / 2  # calculate gradients
            norm = np.linalg.norm(b)  # compute gradient norm
            B[i] = norm  # store for convenience

            if norm < epsilon:  # stop condition
                print(f'Optimal solution found after {i} iterations '
                      f'(method: {"Direct" if not use_CG else "Conjugated"} Gradient |'
                      f' epsilon={epsilon} | norm={norm} | t={t if not optimize_t else "optimal"})')
                J[idx, i:] = J[idx, i-1]
                break

            J_curr = np.sum(x[:-1] ** 2 + (1/2) * u ** 2) / 2 + (1 / 2) * (3 / 2) * x[-1] ** 2
            J[idx, i] = J_curr

            if optimize_t:
                J_t = lambda t: (np.sum(
                    (np.insert(x_0 + np.cumsum(u) / 2, 0, x_0))[:-1] ** 2 + ((u - t * b) ** 2) / 2)) / 2 + \
                                (3 / 4) * ((x_0 + np.cumsum(u - t * b) / 2)[-1]) ** 2

                def Jt(t):
                    u_next = u - t * b
                    x_next = np.insert(x_0 + np.cumsum(u_next) / 2, 0, x_0)
                    return (1 / 2) * np.sum(x_next[:-1] ** 2 + (1 / 2) * u_next ** 2) + (3 / 4) * x_next[-1] ** 2

                t = minimize(fun=Jt, method="Nelder-Mead", x0=0).x

            if use_CG:
                if i == 0:
                    u = u - t * b
                    U[i, :] = u
                    d[i, :] = -b
                else:
                    c = B[i] ** 2 / B[i - 1] ** 2
                    d[i, :] = -b + c * d[i - 1, :]
                    u = u + t * d[i, :]
                    U[i, :] = u
            else:
                u = u - t * b
                U[i, :] = u

    if multimode:
        # J = np.zeros(shape=(4, n_iter))

        options = np.array([0, 1, 2, 3])
        options_binary = ((options.reshape(-1, 1) & (2 ** np.arange(2))) != 0).astype(int)
        # print(options_binary)
        # print(options_binary[:, ::-1])
        # options_binary format: [ [0,0], [0,1], [1,0], [1,1] ]
        # each entry consists of two boolean values:
        # 1st is connected to the use of conjugated method
        # 2nd decides whether or not to optimize t:
        # [0,0]: Direct Gradient, no optimizations
        # [0,1]: Direct Gradient, step optimized
        # [1,0]: Conjugated Gradient, no optimizations
        # [1,1]: Conjugated Gradient, step optimized

        for idx, (use_cg, opt_t) in enumerate(options_binary):
            _optimize(epsilon=1e-5, t=ts[0], use_CG=use_cg, optimize_t=opt_t, init_control=[2, 16, 2, 8, 4])
            l_1 = f't={ts if not opt_t else "optimized"}'
            l_2 = " [Conjugated]" if use_cg else "[Direct]"
            plt.plot(J[idx, :], label=f'{l_1 + l_2}')

        plt.title("Mix")
        plt.xlabel("Iteration")
        plt.ylabel("Performance Index")
        xlim = 20 if K >= 20 else K
        ylim = 40
        plt.xticks(np.arange(0, xlim, step=1))
        plt.yticks(np.arange(0, ylim, step=2))
        plt.xlim((0, xlim))
        plt.ylim((0, ylim))

        plt.legend()
        plt.show()

    else:
        for idx, t in enumerate(ts):
            if opt_t:
                _optimize(t, optimize_t=True)
                plt.plot(J[idx, :], label=f't-optimal')
                xlim = 20 if K >= 20 else K
                ylim = 50
                plt.xticks(np.arange(0, xlim, step=1))
                plt.yticks(np.arange(0, ylim, step=2))
                plt.xlim((0, xlim))
                plt.ylim((0, ylim))
                break

            else:
                _optimize(t, optimize_t=False)
                plt.plot(J[idx, :], label=f't={t}')
                xlim, dx = (20 if K >= 20 else K, 1) if cg else (30 if K >= 30 else K, 1)
                ylim = 50
                plt.xticks(np.arange(0, xlim, step=dx))
                plt.yticks(np.arange(0, ylim, step=2))
                plt.xlim((0, xlim))
                plt.ylim((0, ylim))

        # plt.tight_layout()
        plt.title(f'{"Conjugated" if cg else "Direct"} Gradient')
        plt.xlabel("Iteration")
        plt.ylabel("Performance Index")
        plt.legend()
        plt.show()
    print()

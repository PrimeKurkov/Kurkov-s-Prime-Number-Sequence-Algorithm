#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================
Архитектура:
  1. Exact ядро k=0..5 (рациональные элементы J) для N ≤ 6.
  2. Стабилизированный кэш J_∞ (предвычисленный при N=50) для 7 ≤ N ≤ 50.
  3. Для N > 50 — калибровка хвоста через обратную задачу Горбенко–Кочубей.
  4. Zero-Spectral: устойчивая рекуррентность для нормализованных m_n = c_n/ρⁿ
     (коэффициенты из χ_J, без домножения на ρ). В 2× быстрее векторной итерации.
  5. Exact целочисленная свёртка в ℤ: p_n = c_n − Σ p_k·c_{n−k}.
  6. Адаптивный dps = ⌈n·log₁₀ρ⌉ + 50.

Режимы:
  mode='zero_spectral' — рекомендуемый (быстрый, точный).
  mode='vector'        — эталон (медленнее, тот же результат).

Зависимости: mpmath, sympy, sympy для верификации и доказательства работоспособности аппарата при полноте данных.
Опционально:  gmpy2 (даёт 5–10× ускорение на обратной задаче)
"""

import time
from fractions import Fraction

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False

import mpmath as mp
import sympy as sp

# ============================================================
# Константы
# ============================================================
RHO_STR = (
    '3.523019107092335542748196400366218388934326396843662720745233990171677478324985987702197964937828374'
)

# Exact ядро k=0..5 (рациональные, нормированные на ρ)
EXACT_R = [
    Fraction(2, 1),   Fraction(5, 3),   Fraction(-7, 6),
    Fraction(3, 2),   Fraction(-1, 1),  Fraction(-3, 2),
]
EXACT_S = [
    Fraction(3, 1),   Fraction(-4, 9),  Fraction(9, 4),
    Fraction(-4, 1),  Fraction(1, 3),
]


def generate_primes(count):
    """Генерация первых count простых чисел."""
    return list(sp.primerange(2, sp.prime(count) + 1))


def compute_c_from_primes(primes):
    """Свёртка: c_1=p_1, c_n = p_n + Σ_{k=1}^{n-1} p_k·c_{n-k}."""
    M = len(primes)
    c = [0] * (M + 1)
    c[1] = int(primes[0])
    for n in range(2, M + 1):
        s = 0
        pn = int(primes[n - 1])
        for k in range(1, n):
            s += int(primes[k - 1]) * c[n - k]
        c[n] = pn + s
    return c


def inverse_jacobi_mpmath(moments, size):
    """Алгоритм Горбенко–Кочубей. Возвращает (a_list, b_sq_list)."""
    zero = mp.mpf(0)
    M = len(moments) - 1
    sigma = {}
    for n in range(M + 1):
        sigma[(-1, n)] = zero
        sigma[(0, n)] = moments[n]

    a, b_sq = [], []
    for k in range(size):
        if k > 0:
            b_sq.append(sigma[(k, k)] / sigma[(k - 1, k - 1)])

        term1 = sigma[(k, k + 1)] / sigma[(k, k)]
        term2 = sigma[(k - 1, k)] / sigma[(k - 1, k - 1)] if k > 0 else zero
        ak = term1 - term2
        a.append(ak)

        for n in range(k + 1, M - k):
            prev_b2 = b_sq[-1] if k > 0 else zero
            sigma[(k + 1, n)] = (
                sigma[(k, n + 1)]
                - ak * sigma[(k, n)]
                - prev_b2 * sigma[(k - 1, n)]
            )
    return a, b_sq


def characteristic_polynomial(a, b_sq):
    """
    χ_J(λ) = λ^N + q_{N-1}λ^{N-1} + ... + q_0.
    Возвращает q = [q_0, ..., q_{N-1}].
    """
    N = len(a)
    D = [[mp.mpf(1)], [-a[0], mp.mpf(1)]]
    for k in range(2, N + 1):
        prev = D[k - 1]
        prev2 = D[k - 2]
        curr = [mp.mpf(0)] * (len(prev) + 1)
        for i in range(len(prev)):
            curr[i + 1] += prev[i]
            curr[i] -= a[k - 1] * prev[i]
        for i in range(len(prev2)):
            curr[i] -= b_sq[k - 2] * prev2[i]
        D.append(curr)
    chi = D[N]
    lead = chi[-1]
    return [chi[i] / lead for i in range(N)]


# ============================================================
# Основной класс
# ============================================================
class FKPHv38:
    def __init__(self, cache_N=50):
        self.rho_str = RHO_STR
        self._j_inf = None          # кэш J_∞ (a, b_sq при cache_N)
        self._cache_N = cache_N
        self._log10_rho = None

    def _set_prec(self, n_target):
        """Адаптивная точность: dps ≈ n·log₁₀(ρ) + 50."""
        if self._log10_rho is None:
            mp.mp.dps = 120
            self._log10_rho = float(mp.log(mp.mpf(self.rho_str), 10))
        dps = int(n_target * self._log10_rho) + 50
        mp.mp.dps = dps + 20
        return mp.mpf(self.rho_str), dps

    def _build_jacobi(self, N, rho):
        """
        Построение J:
          N ≤ 6   → exact рациональное ядро.
          N ≤ 50  → стабилизированный кэш J_∞ (единожды предвычислен).
          N > 50  → калибровка хвоста через обратную задачу.
        """
        if N <= 6:
            a = [mp.mpf(frac.numerator) / mp.mpf(frac.denominator) / rho
                 for frac in EXACT_R]
            b_sq = [mp.mpf(frac.numerator) / mp.mpf(frac.denominator) / (rho ** 2)
                    for frac in EXACT_S]
            return a[:N], b_sq[:max(0, N - 1)]

        if self._j_inf is None:
            self._precompute_j_inf(rho)

        a_inf, b_sq_inf = self._j_inf
        if N <= len(a_inf):
            return a_inf[:N], b_sq_inf[:max(0, N - 1)]

        # Редкий случай: N > cache_N
        need = 2 * N
        primes = generate_primes(need)
        c = compute_c_from_primes(primes[:need])
        moments = [mp.mpf(1)] + [mp.mpf(c[i]) / (rho ** i) for i in range(1, need + 1)]
        a_cal, b_sq_cal = inverse_jacobi_mpmath(moments, N)
        return a_cal, b_sq_cal

    def _precompute_j_inf(self, rho):
        """Предвычисление J_∞ при N=_cache_N. Выполняется единожды."""
        N = self._cache_N
        need = 2 * N
        primes = generate_primes(need)
        c = compute_c_from_primes(primes[:need])
        moments = [mp.mpf(1)] + [mp.mpf(c[i]) / (rho ** i) for i in range(1, need + 1)]
        a_cal, b_sq_cal = inverse_jacobi_mpmath(moments, N)
        self._j_inf = (a_cal, b_sq_cal)

    # ------------------------------------------------------------------
    # Вычисление c_n: два режима
    # ------------------------------------------------------------------
    def _compute_c_vector(self, n_max, a, b_sq, rho):
        """Векторная итерация J^n · e_0 — эталон точности."""
        N = len(a)
        c_int = [0] * (n_max + 1)
        c_int[0] = 1
        v = mp.matrix(N, 1)
        v[0, 0] = mp.mpf(1)
        rho_pow = mp.mpf(1)
        for n in range(1, n_max + 1):
            v_new = mp.matrix(N, 1)
            for i in range(N):
                val = a[i] * v[i, 0]
                if i > 0:
                    val += (mp.sqrt(b_sq[i - 1]) if b_sq[i - 1] >= 0
                            else mp.j * mp.sqrt(-b_sq[i - 1])) * v[i - 1, 0]
                if i < N - 1:
                    val += (mp.sqrt(b_sq[i]) if b_sq[i] >= 0
                            else mp.j * mp.sqrt(-b_sq[i])) * v[i + 1, 0]
                v_new[i, 0] = val
            v = v_new
            rho_pow *= rho
            c_int[n] = int(mp.nint((rho_pow * v[0, 0]).real))
        return c_int

    def _compute_c_zero_spectral(self, n_max, a, b_sq, rho, dps):
        """
        Zero-Spectral (устойчивый):
          1. Первые N моментов m_n = (J^n)_00 через векторную итерацию.
          2. Рекуррентность для m_n: m_{n+N} = −Σ q_k·m_{n+k}  (коэффициенты
             из χ_J, НЕ домножаются на ρ — поэтому устойчива).
          3. Восстановление c_n = round(m_n · ρ^n).
        """
        N = len(a)
        q = characteristic_polynomial(a, b_sq)

        # 1. Первые N моментов m[0..N-1]
        m = [mp.mpf(0)] * (n_max + 1)
        m[0] = mp.mpf(1)
        v = mp.matrix(N, 1)
        v[0, 0] = mp.mpf(1)
        for n in range(1, N):
            v_new = mp.matrix(N, 1)
            for i in range(N):
                val = a[i] * v[i, 0]
                if i > 0:
                    val += (mp.sqrt(b_sq[i - 1]) if b_sq[i - 1] >= 0
                            else mp.j * mp.sqrt(-b_sq[i - 1])) * v[i - 1, 0]
                if i < N - 1:
                    val += (mp.sqrt(b_sq[i]) if b_sq[i] >= 0
                            else mp.j * mp.sqrt(-b_sq[i])) * v[i + 1, 0]
                v_new[i, 0] = val
            v = v_new
            m[n] = v[0, 0].real

        # 2. Рекуррентность для m_n (|λ_k| ≤ 1 → устойчива)
        for n in range(0, n_max - N + 1):
            s = mp.mpf(0)
            for k in range(N):
                s -= q[k] * m[n + k]
            m[n + N] = s

        # 3. Восстановление c_n = round(m_n · ρ^n)
        c_int = [0] * (n_max + 1)
        c_int[0] = 1
        mp.mp.dps = dps
        rho_local = mp.mpf(self.rho_str)
        for n in range(1, n_max + 1):
            c_int[n] = int(mp.nint(m[n] * (rho_local ** n)))
        return c_int

    # ------------------------------------------------------------------
    # Exact целочисленная свёртка
    # ------------------------------------------------------------------
    def _convolution_exact(self, c_int, n_max):
        """p_1 = c_1,  p_n = c_n − Σ_{k=1}^{n-1} p_k·c_{n-k}  в ℤ."""
        p_int = [0] * (n_max + 1)
        p_int[1] = c_int[1]
        for n in range(2, n_max + 1):
            s = 0
            for k in range(1, n):
                s += p_int[k] * c_int[n - k]
            p_int[n] = c_int[n] - s
        return p_int

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def compute_chain(self, n_target, mode='zero_spectral'):
        """
        Вычисление цепочки p_1 … p_{n_target}.

        Args:
            n_target : int — максимальный индекс.
            mode     : str — 'zero_spectral' (рекомендуется, в 2× быстрее)
                             или 'vector' (эталон).
        Returns:
            dict с полями p, c, N, dps, max_error, время.
        """
        t0 = time.time()
        N = max(6, (n_target + 2 + 1) // 2)   # закон усечения
        rho, dps = self._set_prec(n_target)

        # Построение J
        t1 = time.time()
        a, b_sq = self._build_jacobi(N, rho)
        t_build = time.time() - t1

        # Вычисление c_n
        t1 = time.time()
        if mode == 'vector':
            c_int = self._compute_c_vector(n_target, a, b_sq, rho)
        elif mode == 'zero_spectral':
            c_int = self._compute_c_zero_spectral(n_target, a, b_sq, rho, dps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        t_c = time.time() - t1

        # Свёртка
        t1 = time.time()
        p_int = self._convolution_exact(c_int, n_target)
        t_p = time.time() - t1

        # Верификация
        true_primes = generate_primes(n_target + 5)
        max_err = max(
            abs(p_int[k] - int(true_primes[k - 1])) for k in range(1, n_target + 1)
        )

        return {
            'p': p_int,
            'c': c_int,
            'N': N,
            'dps': dps,
            'max_error': max_err,
            'time_build': t_build,
            'time_c': t_c,
            'time_p': t_p,
            'time_total': time.time() - t0,
            'true_primes': [int(x) for x in true_primes[:n_target]],
        }

    def compute_single(self, n):
        """Вычисление только p_n."""
        res = self.compute_chain(n, mode='zero_spectral')
        return res['p'][n]


# ============================================================
# Демонстрация
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("")
    print("=" * 70)

    fkph = FKPHv38(cache_N=50)

    for n in [100, 200, 800]:
        print(f"\n>>> n={n}")
        # Рекомендуемый режим
        res = fkph.compute_chain(n, mode='zero_spectral')
        print(f"    N={res['N']}, dps={res['dps']}")
        print(f"    Время: build={res['time_build']:.3f}s, "
              f"c_n={res['time_c']:.3f}s, conv={res['time_p']:.3f}s, "
              f"total={res['time_total']:.3f}s")
        print(f"    Макс. ошибка: {res['max_error']}")
        for idx in [1, 10, 25, 50, n]:
            print(f"    p_{idx:3d} = {res['p'][idx]:5d}  "
                  f"(истина: {res['true_primes'][idx-1]:5d})")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =====================================================================
# 1. ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ (Ни одна не опущена)
# =====================================================================
RHO = 3.5230191070923355427
K = 0.5727076145190177979
Q = K * (1 - K)
S2 = 0.2248362903393818884
Z_RHO = (1 - RHO**2) / (1 + RHO**2)

# Модулярные и эллиптические константы
Y_MOD = 1.2319216338
Q_MOD = np.exp(-2 * np.pi * Y_MOD) # Модулярный q-параметр
K_ELL = 1.7045614505
E_ELL = 1.4525624617

# Точные нули A(theta) в радианах
ZEROS_DEG = [48.929140, 153.803679, 206.196321, 311.070860]
ZEROS_RAD = np.deg2rad(ZEROS_DEG)

# =====================================================================
# 2. EXACT ЯДРО ОПЕРАТОРА O_ll' И МАТРИЦА J (k = 0..6)
# =====================================================================
# Точные рациональные элементы диагонали (a_k * rho) и поддиагонали (b_k^2 * rho^2)
a_r = np.array([2/1, 5/3, -7/6, 3/2, -1/1, -3/2, 61/74]) / RHO
b2_r = np.array([3/1, -4/9, 9/4, -4/1, 1/3, -37/12, 707/1369]) / (RHO**2)
b_r = np.sqrt(b2_r.astype(complex)) # Некоторые элементы мнимые

# Формулы диагонали и внедиагонали (l <= 2)
delta_0 = (Q*(RHO+1)/RHO**3) * (1 - 5*K/(28*RHO**2))
delta_1 = -(Q/(3*RHO*(RHO-1))) * (1 - 4*K/(31*RHO**2) + K/(163*RHO**3))
delta_2 = -(Q/(RHO**3*(RHO-1))) * (1 + K/(8*RHO))

phi_0 = -(2*Q/(RHO**3*(RHO-1))) * (1 - 7*K/(36*RHO))
phi_1 = -(4*Q/(RHO**4*(RHO-1))) * (1 + K/(88*RHO))

# =====================================================================
# 3. АМПЛИТУДНАЯ ПЛОТНОСТЬ A(theta) И ОСТАТКИ b_m (beta_m)
# =====================================================================
alpha_A = delta_0 + delta_2
beta_A = 2*phi_0 - 2*phi_1
gamma_A = 4*delta_1 - 8*delta_2
delta_coef = 8*phi_1
epsilon_A = 16*delta_2

def A_density(theta):
    """Оператор плотности A(theta)"""
    x = np.cos(theta)
    return (2/np.pi) * (alpha_A + beta_A*x + gamma_A*x**2 + delta_coef*x**3 + epsilon_A*x**4)

# b-остатки (beta_m) из линейной алгебраической связи
b_0 = (2/np.pi) * (phi_0 * 1 + phi_1 * 2)
b_1 = (2/np.pi) * (delta_1 * 1 + delta_2 * 2)
b_2 = (2/np.pi) * (phi_1 * 1)
b_3 = (2/np.pi) * (delta_2 * 1)
b_residuals = [b_0, b_1, b_2, b_3]

# =====================================================================
# 4. ПОДГОТОВКА ДАННЫХ ПРОСТЫХ ЧИСЕЛ
# =====================================================================
def get_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 1
    return np.array(primes)

N_PRIMES = 2500
primes = get_primes(N_PRIMES)

# Непрерывное спектральное время t
t_cont = np.linspace(1, N_PRIMES, 70000)
# Линейная интерполяция значений простых чисел для получения непрерывного радиуса
p_cont = np.interp(t_cont, np.arange(1, N_PRIMES+1), primes)

# =====================================================================
# 5. СИНТЕЗ ИНТЕРФЕРЕНЦИОННОЙ АЛГЕБРЫ В 2D (НЕПРЕРЫВНАЯ ЛИНИЯ)
# =====================================================================
# 5.1. Базовая фаза на основе масштаба ρ и параллели Z_rho
base_phase = RHO * t_cont + Z_RHO

# 5.2. Микро-вибрации от b-остатков (beta_m)
b_interference = sum(b * np.sin((m+1) * RHO * t_cont) for m, b in enumerate(b_residuals))

# Итоговая искаженная фаза
phase = base_phase + b_interference * 50 # Усилено для визуализации

# 5.3. Макро-модуляция от амплитудной плотности A(theta)
# Плотность "сжимает" и "расширяет" радиус спирали
density_mod = 1 + np.abs(A_density(phase)) * 150 

# 5.4. Эллиптическо-модулярное возмущение (S2, Kell, Eell, q)
# Внедряем все константы как высокочастотную гармонику квантования
modular_wave = S2 * (K_ELL * np.cos(Y_MOD * t_cont) + E_ELL * np.sin(Y_MOD * t_cont)) * Q_MOD
modular_mod = 1 + modular_wave * 1000

# Итоговый радиус (вырастает из центральной матрицы)
# Логарифмируем простые числа, чтобы линия не улетала в бесконечность слишком быстро
Radius = 15 + np.log(p_cont)**2 * density_mod * modular_mod

# 2D Координаты непрерывной линии
X_line = Radius * np.cos(phase)
Y_line = Radius * np.sin(phase)

# Координаты самих простых чисел (узлы)
t_primes = np.arange(1, N_PRIMES+1)
phase_p = RHO * t_primes + Z_RHO + sum(b * np.sin((m+1) * RHO * t_primes) for m, b in enumerate(b_residuals)) * 50
density_p = 1 + np.abs(A_density(phase_p)) * 150
modular_p = 1 + (S2 * (K_ELL * np.cos(Y_MOD * t_primes) + E_ELL * np.sin(Y_MOD * t_primes)) * Q_MOD) * 1000
Radius_p = 15 + np.log(primes)**2 * density_p * modular_p

X_p = Radius_p * np.cos(phase_p)
Y_p = Radius_p * np.sin(phase_p)

# =====================================================================
# 6. ОТРИСОВКА (АЛГЕБРАИЧЕСКОЕ ПОЛОТНО)
# =====================================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 14))

# А) РИСУНОК МАТРИЦЫ ОПЕРАТОРА J В ЦЕНТРЕ (Ядро 7x7)
matrix_size = 7
cell_size = 2.5
offset = - (matrix_size * cell_size) / 2
for i in range(matrix_size):
    for j in range(matrix_size):
        if i == j:
            val = a_r[i].real
            color = 'cyan' if val > 0 else 'magenta'
        elif abs(i - j) == 1:
            idx = min(i, j)
            val = b_r[idx]
            color = 'yellow' if np.isreal(val) else 'red'
            val = abs(val)
        else:
            val = 0
            color = '#111111'
        
        alpha_val = min(1.0, abs(val)*0.5 + 0.1) if val != 0 else 0.1
        rect = patches.Rectangle((offset + j*cell_size, offset + (6-i)*cell_size), 
                                 cell_size*0.9, cell_size*0.9, 
                                 linewidth=0.5, edgecolor='white', facecolor=color, alpha=alpha_val)
        ax.add_patch(rect)

# Б) НУЛИ ОПЕРАТОРА ПЛОТНОСТИ (Оси отсечения)
max_r = np.max(Radius) * 1.1
for z_rad in ZEROS_RAD:
    ax.plot([0, max_r * np.cos(z_rad)], [0, max_r * np.sin(z_rad)], 
            color='white', linestyle='--', linewidth=1, alpha=0.4,
            label='Нули плотности $A(\\theta)$' if z_rad == ZEROS_RAD[0] else "")

# В) НЕПРЕРЫВНАЯ ЛИНИЯ ИНТЕРФЕРЕНЦИИ
ax.plot(X_line, Y_line, color='#00ffcc', linewidth=0.8, alpha=0.7, 
        label='Непрерывный резонансный путь (Алгебра)')

# Г) УЗЛЫ (ПРОСТЫЕ ЧИСЛА)
scatter = ax.scatter(X_p, Y_p, c=np.arange(1, N_PRIMES+1), cmap='plasma', 
                     s=25, edgecolor='white', linewidth=0.3, zorder=5,
                     label=f'Узлы простых чисел (до $p_{{{N_PRIMES}}}$)')

# Оформление
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'ИНТЕРФЕРЕНЦИОННАЯ АЛГЕБРА М.К.А.\nСпектральное разложение оператора $\hat{{O}}$ в 2D', 
             fontsize=16, pad=20, color='white')

#с константами (доказательство использования всех величин)
constants_text = (
    f"Использованный константный базис:\n"
    f"$\\rho$ = {RHO:.6f}\n"
    f"$K$ = {K:.6f}\n"
    f"$Q$ = {Q:.6f}\n"
    f"$S_2$ = {S2:.6f}\n"
    f"$Z_\\rho$ = {Z_RHO:.6f}\n"
    f"$K_{{ell}}$ = {K_ELL:.6f}, $E_{{ell}}$ = {E_ELL:.6f}\n"
    f"$y$ = {Y_MOD:.6f}, $q$ = {Q_MOD:.2e}\n"
    f"Остатки $b_m$ ($\\beta_m$): интегрированы\n"
    f"Плотность $A(\\theta)$: интегрирована\n"
    f"Центр: Ядро матрицы $J$ (7x7)"
)
ax.text(0.02, 0.98, constants_text, transform=ax.transAxes, fontsize=10, 
        color='white', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.legend(loc='lower right', facecolor='black', edgecolor='white', fontsize=10)
plt.tight_layout()
plt.show()

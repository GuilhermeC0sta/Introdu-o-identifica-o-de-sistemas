import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import control
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Caminhos relativos ao diretório onde o script está localizado
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
IMG_DIR  = BASE_DIR / 'images'

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})
T = 0.1  # Período de amostragem
np.random.seed(42)

# ============================================================
# QUESTÃO 1 — Sistemas contínuos e discretização
# ============================================================

# a) Ga(s) = (0.5s^2 + 2s + 2) / (s^3 + 3s^2 + 4s + 2)
num_a = [0.5, 2, 2]
den_a = [1, 3, 4, 2]
Ga = control.tf(num_a, den_a)

# b) Gb(s) = 2.5 / (s^2 + s + 2.5)
num_b = [2.5]
den_b = [1, 1, 2.5]
Gb = control.tf(num_b, den_b)

print("=" * 60)
print("QUESTÃO 1 — SISTEMAS CONTÍNUOS")
print("=" * 60)
print("\nGa(s) =", Ga)
print("Polos de Ga:", control.poles(Ga))
print("\nGb(s) =", Gb)
print("Polos de Gb:", control.poles(Gb))

# Discretização ZOH
Ga_d = control.sample_system(Ga, T, method='zoh')
Gb_d = control.sample_system(Gb, T, method='zoh')

print("\n--- Ga(z) [ZOH, T=0.1s] ---")
print(Ga_d)
print("Polos de Ga_d:", control.poles(Ga_d))

print("\n--- Gb(z) [ZOH, T=0.1s] ---")
print(Gb_d)
print("Polos de Gb_d:", control.poles(Gb_d))

# Extrair coeficientes para equação a diferenças
def tf_to_difference_eq(G_d, name):
    num = np.array(G_d.num[0][0])
    den = np.array(G_d.den[0][0])
    # Normalizar pelo coeficiente líder do denominador
    num = num / den[0]
    den = den / den[0]
    print(f"\nEquação a diferenças de {name}:")
    n = len(den) - 1
    terms_y = " + ".join([f"({-den[i+1]:.6f})*y(k-{i+1})" for i in range(n)])
    terms_u = " + ".join([f"({num[i]:.6f})*u(k-{i})" for i in range(len(num))])
    print(f"y(k) = {terms_y} + {terms_u}")
    return num, den

num_ad, den_ad = tf_to_difference_eq(Ga_d, "Ga(z)")
num_bd, den_bd = tf_to_difference_eq(Gb_d, "Gb(z)")

# Simulação degrau
t_cont = np.linspace(0, 10, 1000)
t_disc = np.arange(0, 10, T)
u_step = np.ones_like(t_cont)
u_step_d = np.ones(len(t_disc))

t_a, y_a = control.step_response(Ga, T=t_cont)
t_b, y_b = control.step_response(Gb, T=t_cont)

# Simular discreto via convolução (lsim)
t_ad, y_ad = control.forced_response(Ga_d, T=t_disc, U=u_step_d)
t_bd, y_bd = control.forced_response(Gb_d, T=t_disc, U=u_step_d)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Questão 1 — Resposta ao Degrau: Contínuo vs Discreto (T=0.1s)", fontweight='bold')

axes[0].plot(t_a, y_a, 'b-', label='Contínuo', linewidth=2)
axes[0].step(t_ad, y_ad, 'r--', label='Discreto ZOH', linewidth=1.5, where='post')
axes[0].set_title("Sistema a)  Ga(s) = (0.5s²+2s+2)/(s³+3s²+4s+2)")
axes[0].set_xlabel("Tempo (s)"); axes[0].set_ylabel("Saída")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_b, y_b, 'b-', label='Contínuo', linewidth=2)
axes[1].step(t_bd, y_bd, 'r--', label='Discreto ZOH', linewidth=1.5, where='post')
axes[1].set_title("Sistema b)  Gb(s) = 2.5/(s²+s+2.5)")
axes[1].set_xlabel("Tempo (s)"); axes[1].set_ylabel("Saída")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'q1_step_response.png', bbox_inches='tight')
plt.close()
print("\n[Fig. salva] q1_step_response.png")

# ============================================================
# QUESTÃO 2 — Mínimos Quadrados + Ruído
# ============================================================
print("\n" + "=" * 60)
print("QUESTÃO 2 — MÍNIMOS QUADRADOS")
print("=" * 60)

def simulate_difference_eq(num, den, u_input):
    """Simula equação a diferenças y(k) = -sum(den[i+1]*y(k-i-1)) + sum(num[j]*u(k-j))"""
    N = len(u_input)
    y = np.zeros(N)
    na = len(den) - 1
    nb = len(num) - 1
    for k in range(N):
        for i in range(1, na + 1):
            if k - i >= 0:
                y[k] -= den[i] * y[k - i]
        for j in range(nb + 1):
            if k - j >= 0:
                y[k] += num[j] * u_input[k - j]
    return y

N = 100
# Entrada degrau
u_step100 = np.ones(N)
# Entrada uniforme [-1,1] média zero
u_unif = np.random.uniform(-1, 1, N)
u_unif -= u_unif.mean()  # garantir média zero

def build_regressor(y, u, na, nb):
    """Constrói matriz de regressores Phi para ARX de ordem (na, nb)"""
    N = len(y)
    n_params = na + nb + 1
    Phi = np.zeros((N, n_params))
    for k in range(N):
        row = []
        for i in range(1, na + 1):
            row.append(-y[k - i] if k - i >= 0 else 0.0)
        for j in range(nb + 1):
            row.append(u[k - j] if k - j >= 0 else 0.0)
        Phi[k, :] = row
    return Phi

def least_squares(Phi, y):
    """Estimador de Mínimos Quadrados: theta = (Phi'*Phi)^{-1} * Phi' * y"""
    return np.linalg.lstsq(Phi, y, rcond=None)[0]

def compute_residuals(y, Phi, theta):
    y_hat = Phi @ theta
    return y - y_hat

def emq_analysis(num_d, den_d, sys_name):
    """Executa análise EMQ para um sistema dado"""
    # Gerar dados com entrada degrau e uniforme
    y_step = simulate_difference_eq(num_d, den_d, u_step100)
    y_unif = simulate_difference_eq(num_d, den_d, u_unif)

    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    fig.suptitle(f"Q2 — Resíduos EMQ — {sys_name}", fontweight='bold', fontsize=13)

    results = {}
    for order in range(1, 6):
        na = order; nb = order
        # Degrau
        Phi_s = build_regressor(y_step, u_step100, na, nb)
        theta_s = least_squares(Phi_s, y_step)
        res_s = compute_residuals(y_step, Phi_s, theta_s)
        # Uniforme
        Phi_u = build_regressor(y_unif, u_unif, na, nb)
        theta_u = least_squares(Phi_u, y_unif)
        res_u = compute_residuals(y_unif, Phi_u, theta_u)

        ax = axes[order - 1]
        ax[0].plot(res_s, 'b-', alpha=0.7, label='Resíduo')
        ax[0].axhline(res_s.mean(), color='r', linestyle='--', label=f'Média={res_s.mean():.4f}')
        ax[0].set_title(f"Ordem {order} — Entrada Degrau")
        ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
        ax[0].set_ylabel("Resíduo")

        ax[1].plot(res_u, 'g-', alpha=0.7, label='Resíduo')
        ax[1].axhline(res_u.mean(), color='r', linestyle='--', label=f'Média={res_u.mean():.4f}')
        ax[1].set_title(f"Ordem {order} — Entrada Uniforme")
        ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)

        results[order] = {'theta_step': theta_s, 'res_step': res_s,
                          'theta_unif': theta_u, 'res_unif': res_u}
        print(f"  [{sys_name}] Ordem {order} | Degrau: mean_res={res_s.mean():.5f}, std={res_s.std():.5f} | "
              f"Unif: mean_res={res_u.mean():.5f}, std={res_u.std():.5f}")

    axes[-1][0].set_xlabel("Amostras")
    axes[-1][1].set_xlabel("Amostras")
    plt.tight_layout()
    tag = sys_name.replace("(", "").replace(")", "").replace(" ", "_").lower()
    fname = IMG_DIR / f'q2_residuals_{tag}.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"[Fig. salva] {fname}")
    return results, y_step, y_unif

print("\n--- Sistema a ---")
res_a, y_step_a, y_unif_a = emq_analysis(num_ad, den_ad, "Sistema a")
print("\n--- Sistema b ---")
res_b, y_step_b, y_unif_b = emq_analysis(num_bd, den_bd, "Sistema b")

# --- Parte com Ruído: 100 identificações de ordem 3 ---
def noise_analysis(num_d, den_d, y_clean_step, y_clean_unif, u_s, u_u, sys_name):
    """100 identificações com ruído dinâmico e de sensor"""
    na = nb = 3
    n_runs = 100
    sigma = 0.05

    params_dyn_s = []; params_sen_s = []
    params_dyn_u = []; params_sen_u = []

    for _ in range(n_runs):
        e_dyn_s = np.random.normal(0, sigma, N)
        e_sen_s = np.random.normal(0, sigma, N)
        # Ruído dinâmico: adicionado a cada passo
        y_dyn_s = simulate_difference_eq(num_d, den_d, u_s) + e_dyn_s
        # Ruído sensor: adicionado ao vetor inteiro
        y_sen_s = y_clean_step + e_sen_s

        Phi_d = build_regressor(y_dyn_s, u_s, na, nb)
        Phi_s2 = build_regressor(y_sen_s, u_s, na, nb)
        params_dyn_s.append(least_squares(Phi_d, y_dyn_s))
        params_sen_s.append(least_squares(Phi_s2, y_sen_s))

        e_dyn_u = np.random.normal(0, sigma, N)
        e_sen_u = np.random.normal(0, sigma, N)
        y_dyn_u = simulate_difference_eq(num_d, den_d, u_u) + e_dyn_u
        y_sen_u = y_clean_unif + e_sen_u

        Phi_du = build_regressor(y_dyn_u, u_u, na, nb)
        Phi_su = build_regressor(y_sen_u, u_u, na, nb)
        params_dyn_u.append(least_squares(Phi_du, y_dyn_u))
        params_sen_u.append(least_squares(Phi_su, y_sen_u))

    params_dyn_s = np.array(params_dyn_s)
    params_sen_s = np.array(params_sen_s)
    params_dyn_u = np.array(params_dyn_u)
    params_sen_u = np.array(params_sen_u)

    print(f"\n  [{sys_name}] Ruído dinâmico (degrau) — média params: {params_dyn_s.mean(axis=0).round(4)}")
    print(f"  [{sys_name}] Ruído dinâmico (degrau) — std   params: {params_dyn_s.std(axis=0).round(4)}")
    print(f"  [{sys_name}] Ruído sensor  (degrau) — média params: {params_sen_s.mean(axis=0).round(4)}")
    print(f"  [{sys_name}] Ruído sensor  (degrau) — std   params: {params_sen_s.std(axis=0).round(4)}")

    n_params = params_dyn_s.shape[1]
    param_labels = [f"a{i+1}" for i in range(na)] + [f"b{i}" for i in range(nb+1)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Q2 — Distribuição dos Parâmetros (100 runs, ordem 3) — {sys_name}", fontweight='bold')

    for j in range(n_params):
        axes[0, 0].plot(params_dyn_s[:, j], alpha=0.6, label=param_labels[j])
        axes[0, 1].plot(params_sen_s[:, j], alpha=0.6, label=param_labels[j])
        axes[1, 0].plot(params_dyn_u[:, j], alpha=0.6, label=param_labels[j])
        axes[1, 1].plot(params_sen_u[:, j], alpha=0.6, label=param_labels[j])

    for ax, title in zip(axes.flat,
                         ["Ruído Dinâmico — Degrau", "Ruído Sensor — Degrau",
                          "Ruído Dinâmico — Uniforme", "Ruído Sensor — Uniforme"]):
        ax.set_title(title); ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("Realização"); ax.set_ylabel("Valor do parâmetro")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tag = sys_name.replace(" ", "_").lower()
    fname = IMG_DIR / f'q2_noise_{tag}.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"[Fig. salva] {fname}")

print("\n--- Análise de Ruído ---")
noise_analysis(num_ad, den_ad, y_step_a, y_unif_a, u_step100, u_unif, "Sistema_a")
noise_analysis(num_bd, den_bd, y_step_b, y_unif_b, u_step100, u_unif, "Sistema_b")

# ============================================================
# QUESTÃO 3 — Validação dos modelos
# ============================================================
print("\n" + "=" * 60)
print("QUESTÃO 3 — VALIDAÇÃO")
print("=" * 60)

def validate_model(y_val, u_val, theta, na, nb, name=""):
    """Validação: SQE, R2, SNR"""
    Phi_val = build_regressor(y_val, u_val, na, nb)
    y_hat = Phi_val @ theta
    res = y_val - y_hat
    sse = np.sum(res ** 2)
    sst = np.sum((y_val - y_val.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0
    snr = 10 * np.log10(np.var(y_val) / np.var(res)) if np.var(res) > 0 else np.inf
    mse = sse / len(y_val)
    return mse, r2, snr

# Validação com nova entrada aleatória
N_val = 100
u_val = np.random.uniform(-1, 1, N_val)
u_val -= u_val.mean()

print("\n--- Sistema a: Validação dos modelos (entrada uniforme, 100 amostras) ---")
print(f"{'Ordem':>6} {'MSE_est':>12} {'MSE_val':>12} {'R²_val':>10} {'SNR_val':>10}")
for order in range(1, 6):
    na = nb = order
    theta = res_a[order]['theta_unif']
    y_val_a = simulate_difference_eq(num_ad, den_ad, u_val)
    mse_est = np.mean(res_a[order]['res_unif']**2)
    mse_v, r2_v, snr_v = validate_model(y_val_a, u_val, theta, na, nb)
    print(f"{order:>6} {mse_est:>12.6f} {mse_v:>12.6f} {r2_v:>10.4f} {snr_v:>10.2f} dB")

print("\n--- Sistema b: Validação dos modelos ---")
print(f"{'Ordem':>6} {'MSE_est':>12} {'MSE_val':>12} {'R²_val':>10} {'SNR_val':>10}")
for order in range(1, 6):
    na = nb = order
    theta = res_b[order]['theta_unif']
    y_val_b = simulate_difference_eq(num_bd, den_bd, u_val)
    mse_est = np.mean(res_b[order]['res_unif']**2)
    mse_v, r2_v, snr_v = validate_model(y_val_b, u_val, theta, na, nb)
    print(f"{order:>6} {mse_est:>12.6f} {mse_v:>12.6f} {r2_v:>10.4f} {snr_v:>10.2f} dB")

# ============================================================
# HELPER — ARX e ARMAX
# ============================================================

def arx_identify(u, y, na, nb):
    """ARX: A(z)y = B(z)u"""
    N = len(y)
    n_params = na + nb
    Phi = np.zeros((N, n_params))
    for k in range(N):
        row = []
        for i in range(1, na + 1):
            row.append(-y[k - i] if k - i >= 0 else 0.0)
        for j in range(1, nb + 1):
            row.append(u[k - j] if k - j >= 0 else 0.0)
        Phi[k, :] = row
    theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    y_hat = Phi @ theta
    mse = np.mean((y - y_hat) ** 2)
    return theta, mse, y_hat

def armax_identify(u, y, na, nb, nc, max_iter=20, tol=1e-6):
    """ARMAX: A(z)y = B(z)u + C(z)e — via iteração de mínimos quadrados"""
    N = len(y)
    n_params = na + nb + nc
    e = np.zeros(N)
    theta_prev = np.zeros(n_params)
    for iteration in range(max_iter):
        Phi = np.zeros((N, n_params))
        for k in range(N):
            row = []
            for i in range(1, na + 1):
                row.append(-y[k - i] if k - i >= 0 else 0.0)
            for j in range(1, nb + 1):
                row.append(u[k - j] if k - j >= 0 else 0.0)
            for l in range(1, nc + 1):
                row.append(e[k - l] if k - l >= 0 else 0.0)
            Phi[k, :] = row
        theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
        y_hat = Phi @ theta
        e = y - y_hat
        if np.max(np.abs(theta - theta_prev)) < tol:
            break
        theta_prev = theta.copy()
    mse = np.mean(e ** 2)
    return theta, mse, y_hat

def mse_validation(u_val, y_val, theta, na, nb, nc=0, model='arx'):
    """Calcula MSE de validação dado um modelo estimado"""
    N = len(y_val)
    e = np.zeros(N)
    Phi = np.zeros((N, na + nb + nc))
    y_hat = np.zeros(N)
    for k in range(N):
        row = []
        for i in range(1, na + 1):
            row.append(-y_hat[k - i] if k - i >= 0 else 0.0)
        for j in range(1, nb + 1):
            row.append(u_val[k - j] if k - j >= 0 else 0.0)
        if nc > 0:
            for l in range(1, nc + 1):
                row.append(e[k - l] if k - l >= 0 else 0.0)
        Phi[k, :] = row
        y_hat[k] = Phi[k, :] @ theta
        e[k] = y_val[k] - y_hat[k]
    return np.mean(e ** 2)

def split_data(data, frac=0.6):
    """Divide dados em estimação e validação"""
    n = int(len(data) * frac)
    return data[:n], data[n:]

# ============================================================
# QUESTÕES 4 — dados_1 e dados_2
# ============================================================
print("\n" + "=" * 60)
print("QUESTÃO 4 — IDENTIFICAÇÃO dados_1 e dados_2")
print("=" * 60)

def identify_dataset(data, ds_name, q_label="Q4"):
    """Testa ARX e ARMAX de ordens 1-5, retorna tabela de MSE"""
    u_all = data[:, 0]
    y_all = data[:, 1]
    n_est = int(0.6 * len(u_all))
    u_e, y_e = u_all[:n_est], y_all[:n_est]
    u_v, y_v = u_all[n_est:], y_all[n_est:]

    print(f"\n{ds_name} — n_est={n_est}, n_val={len(u_all)-n_est}")
    header = f"{'Modelo':>12} {'Ordem':>6} {'MSE_est':>12} {'MSE_val':>12}"
    print(header)
    print("-" * len(header))

    best_mse_val = np.inf
    best_model = None
    table_rows = []

    for order in range(1, 6):
        na = nb = nc = order
        # ARX
        theta_arx, mse_est_arx, _ = arx_identify(u_e, y_e, na, nb)
        mse_val_arx = mse_validation(u_v, y_v, theta_arx, na, nb, 0, 'arx')
        row_arx = (f"{'ARX':>12} {order:>6} {mse_est_arx:>12.6f} {mse_val_arx:>12.6f}")
        print(row_arx)
        table_rows.append({'model': 'ARX', 'order': order,
                           'mse_est': mse_est_arx, 'mse_val': mse_val_arx,
                           'theta': theta_arx, 'na': na, 'nb': nb, 'nc': 0})
        if mse_val_arx < best_mse_val:
            best_mse_val = mse_val_arx
            best_model = table_rows[-1]

        # ARMAX
        theta_armax, mse_est_armax, _ = armax_identify(u_e, y_e, na, nb, nc)
        mse_val_armax = mse_validation(u_v, y_v, theta_armax, na, nb, nc, 'armax')
        row_armax = (f"{'ARMAX':>12} {order:>6} {mse_est_armax:>12.6f} {mse_val_armax:>12.6f}")
        print(row_armax)
        table_rows.append({'model': 'ARMAX', 'order': order,
                           'mse_est': mse_est_armax, 'mse_val': mse_val_armax,
                           'theta': theta_armax, 'na': na, 'nb': nb, 'nc': nc})
        if mse_val_armax < best_mse_val:
            best_mse_val = mse_val_armax
            best_model = table_rows[-1]

    print(f"\n  *** Melhor modelo: {best_model['model']} ordem {best_model['order']} "
          f"| MSE_val={best_model['mse_val']:.6f}")

    # Gráfico comparativo para o melhor modelo
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(f"{q_label} — {ds_name} — Melhor: {best_model['model']} ordem {best_model['order']}",
                 fontweight='bold')

    # Estimação
    bm = best_model
    if bm['model'] == 'ARX':
        _, _, y_hat_e = arx_identify(u_e, y_e, bm['na'], bm['nb'])
    else:
        _, _, y_hat_e = armax_identify(u_e, y_e, bm['na'], bm['nb'], bm['nc'])

    axes[0].plot(y_e, 'b-', label='y medido', alpha=0.7)
    axes[0].plot(y_hat_e, 'r--', label='ŷ estimado', alpha=0.9)
    axes[0].set_title("Conjunto de Estimação")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylabel("Saída")

    # Validação
    N_v = len(y_v)
    y_hat_v = np.zeros(N_v)
    e_v = np.zeros(N_v)
    nc_v = bm['nc']
    for k in range(N_v):
        val = 0
        for i in range(1, bm['na'] + 1):
            val -= bm['theta'][i-1] * (y_hat_v[k-i] if k-i >= 0 else 0.0)
        for j in range(1, bm['nb'] + 1):
            val += bm['theta'][bm['na']+j-1] * (u_v[k-j] if k-j >= 0 else 0.0)
        if nc_v > 0:
            for l in range(1, nc_v + 1):
                val += bm['theta'][bm['na']+bm['nb']+l-1] * (e_v[k-l] if k-l >= 0 else 0.0)
        y_hat_v[k] = val
        e_v[k] = y_v[k] - y_hat_v[k]

    axes[1].plot(y_v, 'b-', label='y medido', alpha=0.7)
    axes[1].plot(y_hat_v, 'r--', label='ŷ validação', alpha=0.9)
    axes[1].set_title("Conjunto de Validação")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Amostras"); axes[1].set_ylabel("Saída")

    plt.tight_layout()
    fname = IMG_DIR / f'{q_label.lower()}_{ds_name.lower().replace(" ","_")}_best_model.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"[Fig. salva] {fname}")
    return table_rows, best_model

data1 = np.loadtxt(DATA_DIR / 'dados_1.txt')
data2 = np.loadtxt(DATA_DIR / 'dados_2.txt')

rows1, best1 = identify_dataset(data1, "dados_1", "Q4")
rows2, best2 = identify_dataset(data2, "dados_2", "Q4")

# ============================================================
# QUESTÃO 5 — dados_3 e dados_4 + variação dos parâmetros
# ============================================================
print("\n" + "=" * 60)
print("QUESTÃO 5 — IDENTIFICAÇÃO dados_3 e dados_4 (parâmetros no tempo)")
print("=" * 60)

def identify_dataset_q5(data, ds_name):
    """Como Q4 mas também plota evolução dos parâmetros no tempo (janela deslizante)"""
    u_all = data[:, 0]
    y_all = data[:, 1]

    # Identificação completa
    table_rows, best = identify_dataset(data, ds_name, "Q5")

    # Análise temporal dos parâmetros com janela deslizante
    best_na = best['na']; best_nb = best['nb']; best_nc = best['nc']
    win = 80
    step_w = 5
    starts = range(0, len(y_all) - win, step_w)
    param_hist = []
    t_hist = []
    for s in starts:
        u_w = u_all[s:s+win]
        y_w = y_all[s:s+win]
        if best['model'] == 'ARX':
            theta_w, _, _ = arx_identify(u_w, y_w, best_na, best_nb)
        else:
            theta_w, _, _ = armax_identify(u_w, y_w, best_na, best_nb, best_nc)
        param_hist.append(theta_w)
        t_hist.append(s + win // 2)

    param_hist = np.array(param_hist)
    n_params = param_hist.shape[1]

    fig, ax = plt.subplots(figsize=(13, 5))
    param_labels = ([f"a{i+1}" for i in range(best_na)] +
                    [f"b{j+1}" for j in range(best_nb)] +
                    ([f"c{l+1}" for l in range(best_nc)] if best_nc > 0 else []))
    for j in range(n_params):
        ax.plot(t_hist, param_hist[:, j], label=param_labels[j] if j < len(param_labels) else f"p{j}")
    ax.set_title(f"Q5 — {ds_name}: Variação dos Parâmetros no Tempo (janela={win})")
    ax.set_xlabel("Amostra central da janela")
    ax.set_ylabel("Valor do parâmetro")
    ax.legend(ncol=3, fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = IMG_DIR / f'q5_{ds_name.lower().replace(" ","_")}_param_time.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"[Fig. salva] {fname}")

    # Verificar variação
    param_std = param_hist.std(axis=0)
    param_mean = param_hist.mean(axis=0)
    variation_ratio = param_std / (np.abs(param_mean) + 1e-8)
    print(f"\n  Variação relativa dos parâmetros ({ds_name}):")
    for j, label in enumerate(param_labels[:n_params]):
        print(f"    {label}: mean={param_mean[j]:.4f}, std={param_std[j]:.4f}, CV={variation_ratio[j]:.3f}")
    if np.any(variation_ratio > 0.3):
        print("  → Parâmetros VARIAM no tempo (sistema possivelmente não estacionário)")
    else:
        print("  → Parâmetros aproximadamente CONSTANTES no tempo")

    return table_rows, best

data3 = np.loadtxt(DATA_DIR / 'dados_3.txt')
data4 = np.loadtxt(DATA_DIR / 'dados_4.txt')
rows3, best3 = identify_dataset_q5(data3, "dados_3")
rows4, best4 = identify_dataset_q5(data4, "dados_4")

# ============================================================
# QUESTÃO 6 — dados_5 e dados_6
#             Critérios Akaike (AIC) e Bayes (BIC)
#             Mínimos Quadrados Recursivo Estendido (EMQR)
# ============================================================
print("\n" + "=" * 60)
print("QUESTÃO 6 — IDENTIFICAÇÃO dados_5 e dados_6 (AIC/BIC + EMQR)")
print("=" * 60)

def aic_bic(mse, n_params, N):
    """AIC e BIC"""
    sse = mse * N
    if sse <= 0:
        sse = 1e-10
    aic = N * np.log(sse / N) + 2 * n_params
    bic = N * np.log(sse / N) + n_params * np.log(N)
    return aic, bic

def recursive_ls_extended(u, y, na, nb, nc, lam=0.98):
    """Mínimos Quadrados Recursivo Estendido (EMQR) com fator de esquecimento lambda"""
    N = len(y)
    n_params = na + nb + nc
    theta = np.zeros(n_params)
    P = 1000 * np.eye(n_params)
    e_hist = np.zeros(N)
    theta_hist = np.zeros((N, n_params))

    for k in range(N):
        phi = np.zeros(n_params)
        for i in range(na):
            phi[i] = -y[k - i - 1] if k - i - 1 >= 0 else 0.0
        for j in range(nb):
            phi[na + j] = u[k - j - 1] if k - j - 1 >= 0 else 0.0
        for l in range(nc):
            phi[na + nb + l] = e_hist[k - l - 1] if k - l - 1 >= 0 else 0.0

        y_hat = phi @ theta
        e = y[k] - y_hat
        e_hist[k] = e

        # Ganho de Kalman
        Pphi = P @ phi
        denom = lam + phi @ Pphi
        K = Pphi / denom
        theta = theta + K * e
        P = (P - np.outer(K, phi @ P)) / lam

        theta_hist[k] = theta

    y_hat_all = np.array([theta_hist[k] @ np.array(
        [-y[k-i-1] if k-i-1 >= 0 else 0.0 for i in range(na)] +
        [u[k-j-1] if k-j-1 >= 0 else 0.0 for j in range(nb)] +
        [e_hist[k-l-1] if k-l-1 >= 0 else 0.0 for l in range(nc)]
    ) for k in range(N)])
    mse = np.mean((y - y_hat_all) ** 2)
    return theta_hist, mse, e_hist, y_hat_all

def identify_q6(data, ds_name):
    u_all = data[:, 0]
    y_all = data[:, 1]
    n_est = int(0.6 * len(u_all))
    u_e, y_e = u_all[:n_est], y_all[:n_est]
    u_v, y_v = u_all[n_est:], y_all[n_est:]
    N_e = len(y_e)

    print(f"\n{ds_name}")
    print(f"{'Modelo':>12} {'Ordem':>6} {'n_params':>9} {'AIC':>10} {'BIC':>10} {'MSE_est':>12} {'MSE_val':>12}")
    print("-" * 75)

    best_aic = np.inf; best_bic = np.inf
    best_aic_model = None; best_bic_model = None
    all_rows = []

    for order in range(1, 6):
        na = nb = nc = order
        # ARX
        theta_arx, mse_arx, _ = arx_identify(u_e, y_e, na, nb)
        n_p_arx = na + nb
        aic_arx, bic_arx = aic_bic(mse_arx, n_p_arx, N_e)
        mse_v_arx = mse_validation(u_v, y_v, theta_arx, na, nb, 0, 'arx')
        print(f"{'ARX':>12} {order:>6} {n_p_arx:>9} {aic_arx:>10.2f} {bic_arx:>10.2f} "
              f"{mse_arx:>12.6f} {mse_v_arx:>12.6f}")
        all_rows.append({'model':'ARX','order':order,'na':na,'nb':nb,'nc':0,
                         'aic':aic_arx,'bic':bic_arx,'mse_est':mse_arx,'mse_val':mse_v_arx,
                         'theta':theta_arx})

        # ARMAX com EMQR
        theta_hist, mse_emqr, e_emqr, y_hat_emqr = recursive_ls_extended(u_e, y_e, na, nb, nc)
        theta_emqr = theta_hist[-1]
        n_p_armax = na + nb + nc
        aic_armax, bic_armax = aic_bic(mse_emqr, n_p_armax, N_e)
        mse_v_armax = mse_validation(u_v, y_v, theta_emqr, na, nb, nc, 'armax')
        print(f"{'ARMAX(EMQR)':>12} {order:>6} {n_p_armax:>9} {aic_armax:>10.2f} {bic_armax:>10.2f} "
              f"{mse_emqr:>12.6f} {mse_v_armax:>12.6f}")
        all_rows.append({'model':'ARMAX','order':order,'na':na,'nb':nb,'nc':nc,
                         'aic':aic_armax,'bic':bic_armax,'mse_est':mse_emqr,'mse_val':mse_v_armax,
                         'theta':theta_emqr,'theta_hist':theta_hist,'e_hist':e_emqr})

        if aic_armax < best_aic:
            best_aic = aic_armax; best_aic_model = all_rows[-1]
        if bic_armax < best_bic:
            best_bic = bic_armax; best_bic_model = all_rows[-1]
        if aic_arx < best_aic:
            best_aic = aic_arx; best_aic_model = all_rows[-2]
        if bic_arx < best_bic:
            best_bic = bic_arx; best_bic_model = all_rows[-2]

    print(f"\n  *** Melhor por AIC: {best_aic_model['model']} ordem {best_aic_model['order']} (AIC={best_aic:.2f})")
    print(f"  *** Melhor por BIC: {best_bic_model['model']} ordem {best_bic_model['order']} (BIC={best_bic:.2f})")

    # Gráfico dos parâmetros EMQR ao longo do tempo para o melhor modelo
    # Pega o ARMAX com menor AIC
    best_armax = next((r for r in all_rows if r['model']=='ARMAX' and r['order']==best_aic_model['order']), all_rows[-1])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"Q6 — {ds_name}: EMQR — Convergência dos Parâmetros e Saída", fontweight='bold')

    th = best_armax.get('theta_hist', None)
    if th is not None:
        n_p = th.shape[1]
        for j in range(n_p):
            axes[0].plot(th[:, j], label=f"θ{j+1}")
        axes[0].set_title(f"Evolução dos Parâmetros EMQR — ARMAX ordem {best_armax['order']}")
        axes[0].set_xlabel("Amostras"); axes[0].set_ylabel("Parâmetro")
        axes[0].legend(ncol=3, fontsize=8); axes[0].grid(True, alpha=0.3)

    # Saída estimada vs medida
    if 'e_hist' in best_armax:
        y_hat_b = y_e - best_armax['e_hist']
        axes[1].plot(y_e, 'b-', alpha=0.7, label='y medido')
        axes[1].plot(y_hat_b, 'r--', alpha=0.9, label='ŷ EMQR')
        axes[1].set_title("Saída estimada vs medida (conjunto de estimação)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel("Amostras"); axes[1].set_ylabel("Saída")

    plt.tight_layout()
    fname = IMG_DIR / f'q6_{ds_name.lower().replace(" ","_")}_emqr.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"[Fig. salva] {fname}")
    return all_rows, best_aic_model

data5 = np.loadtxt(DATA_DIR / 'dados_5.txt')
data6 = np.loadtxt(DATA_DIR / 'dados_6.txt')
rows5, best5 = identify_q6(data5, "dados_5")
rows6, best6 = identify_q6(data6, "dados_6")

print("\n" + "=" * 60)
print("CONCLUÍDO — Todos os gráficos salvos em images/")
print("=" * 60)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations_with_replacement
import warnings
warnings.filterwarnings('ignore')


# --- Carregamento de Dados ---

def load_ballbeam():
    data = np.loadtxt('Terceiro Roteiro/data/ballbeam.dat')
    return data[:700, 0], data[:700, 1], data[700:, 0], data[700:, 1], 'Ball & Beam', 0.1

def load_exchanger():
    data = np.loadtxt('Terceiro Roteiro/data/exchanger.dat')
    return data[:2800, 1], data[:2800, 2], data[2800:, 1], data[2800:, 2], 'Heat Exchanger', 1.0

def load_silverbox():
    df = pd.read_csv('Terceiro Roteiro/data/Schroeder80mV.csv')
    V1 = df['V1'].values - np.nanmean(df['V1'].values)
    V2 = df['V2'].values - np.nanmean(df['V2'].values)
    u, y = V1[10584:10584 + 1024], V2[10584:10584 + 1024]
    return u[:716], y[:716], u[716:], y[716:], 'Silverbox', 1 / 51200

def load_wiener_hammerstein():
    df = pd.read_csv('Terceiro Roteiro/data/WienerHammerBenchmark.csv')
    u, y = df['uBenchMark'].values, df['yBenchMark'].values
    return u[:4000], y[:4000], u[4000:6000], y[4000:6000], 'Wiener-Hammerstein', 1 / 51200


# --- Normalização ---

def normalize(y, u):
    my, sy = np.mean(y), np.std(y)
    mu, su = np.mean(u), np.std(u)
    return (y - my) / sy, (u - mu) / su, my, sy, mu, su


# --- Geração de Regressores NARX ---

def build_regressors(y, u, ny, nu, nl):
    N = len(y)
    ml = max(ny, nu)
    n = N - ml
    base, blbl = [], []
    for i in range(1, ny + 1):
        base.append(y[ml - i:N - i])
        blbl.append(f'y{i}')
    for i in range(1, nu + 1):
        base.append(u[ml - i:N - i])
        blbl.append(f'u{i}')
    cols, lbls = [], []
    for deg in range(1, nl + 1):
        for combo in combinations_with_replacement(range(len(base)), deg):
            t = np.ones(n)
            parts = []
            for idx in combo:
                t = t * base[idx]
                parts.append(blbl[idx])
            cols.append(t)
            lbls.append('·'.join(parts))
    return np.column_stack(cols), y[ml:], lbls, ml

def eval_term(lbl, y_buf, u_buf):
    val = 1.0
    for p in lbl.split('·'):
        lag = int(p[1:])
        val *= y_buf[lag - 1] if p[0] == 'y' else u_buf[lag - 1]
    return val


# --- FROLS com MGS ---

def frols_mgs(Psi, y, max_terms=15, tol=1e-4):
    N, M = Psi.shape
    avail, sel, Q, ERR = list(range(M)), [], [], []
    yy = np.dot(y, y) + 1e-12

    for _ in range(min(max_terms, M)):
        best_err, best_i, best_q = -1, -1, None
        for i in avail:
            p = Psi[:, i].copy()
            for q in Q:
                p -= (np.dot(q, p) / (np.dot(q, q) + 1e-14)) * q
            nn = np.dot(p, p)
            if nn < 1e-12:
                continue
            err = np.dot(p, y) ** 2 / (nn * yy)
            if err > best_err:
                best_err, best_i, best_q = err, i, p
        if best_i == -1:
            break
        ERR.append(best_err)
        sel.append(best_i)
        Q.append(best_q)
        avail.remove(best_i)
        if np.sum(ERR) >= 1 - tol:
            break

    ns = len(sel)
    g = np.array([np.dot(Q[i], y) / (np.dot(Q[i], Q[i]) + 1e-14) for i in range(ns)])
    A = np.eye(ns)
    for i in range(ns):
        for j in range(i + 1, ns):
            A[i, j] = np.dot(Q[i], Psi[:, sel[j]]) / (np.dot(Q[i], Q[i]) + 1e-14)
    theta = np.linalg.lstsq(A, g, rcond=None)[0]
    return sel, theta, np.array(ERR)


# --- Simulação Livre ---

def simulate(theta, lbls, u, y, ny, nu, clip=1e4):
    N = len(u)
    ml = max(ny, nu)
    ys = np.zeros(N)
    ys[:ml] = y[:ml]
    for t in range(ml, N):
        y_buf = [ys[t - i] for i in range(1, ny + 1)]
        u_buf = [u[t - i] for i in range(1, nu + 1)]
        yhat = sum(c * eval_term(l, y_buf, u_buf) for c, l in zip(theta, lbls))
        ys[t] = np.clip(yhat, -clip, clip)
    return ys

def rmse_sim(theta, lbls, u, y, ny, nu):
    ml = max(ny, nu)
    ys = simulate(theta, lbls, u, y, ny, nu)
    return np.sqrt(np.nanmean((y[ml:] - ys[ml:]) ** 2))

def ols_ridge(Psi, y, alpha=1e-6):
    PtP = Psi.T @ Psi + alpha * np.eye(Psi.shape[1])
    return np.linalg.solve(PtP, Psi.T @ y)


# --- SEMP ---

def semp(Psi, y_osa, lbls, u_tr, y_tr, ny, nu, n_init=12, tol=1e-4):
    sel, _, ERR = frols_mgs(Psi, y_osa, max_terms=n_init, tol=tol)
    cur_sel = list(sel)
    cur_lbls = [lbls[i] for i in cur_sel]
    theta = ols_ridge(Psi[:, cur_sel], y_osa)
    best_r = rmse_sim(theta, cur_lbls, u_tr, y_tr, ny, nu)

    for _ in range(30):
        if len(cur_lbls) <= 2:
            break
        improved = False
        for i in range(len(cur_lbls)):
            t_sel = [cur_sel[j] for j in range(len(cur_sel)) if j != i]
            t_lbls = [cur_lbls[j] for j in range(len(cur_lbls)) if j != i]
            th_t = ols_ridge(Psi[:, t_sel], y_osa)
            r_t = rmse_sim(th_t, t_lbls, u_tr, y_tr, ny, nu)
            if r_t < best_r:
                best_r, theta = r_t, th_t
                cur_sel, cur_lbls = t_sel, t_lbls
                improved = True
                break
        if not improved:
            break

    return cur_lbls, theta, best_r


# --- Pipeline Principal ---

cfg = {
    'Ball & Beam':        {'ny': 3, 'nu': 3, 'nl': 2, 'mt': 12},
    'Heat Exchanger':     {'ny': 4, 'nu': 4, 'nl': 2, 'mt': 12},
    'Silverbox':          {'ny': 2, 'nu': 2, 'nl': 3, 'mt': 12},
    'Wiener-Hammerstein': {'ny': 3, 'nu': 3, 'nl': 2, 'mt': 12},
}

loaders = {
    'Ball & Beam':        load_ballbeam,
    'Heat Exchanger':     load_exchanger,
    'Silverbox':          load_silverbox,
    'Wiener-Hammerstein': load_wiener_hammerstein,
}

summary = {}

for name, loader in loaders.items():
    u_tr0, y_tr0, u_te0, y_te0, _, Ts = loader()
    p = cfg[name]
    ny, nu, nl, mt = p['ny'], p['nu'], p['nl'], p['mt']

    yn_tr, un_tr, my, sy, mu, su = normalize(y_tr0, u_tr0)
    yn_te = (y_te0 - my) / sy
    un_te = (u_te0 - mu) / su

    Psi, y_osa, lbls, ml = build_regressors(yn_tr, un_tr, ny, nu, nl)

    sel, theta_fr, ERR = frols_mgs(Psi, y_osa, max_terms=mt, tol=1e-4)
    sel_lbls = [lbls[i] for i in sel]
    theta_fr = ols_ridge(Psi[:, sel], y_osa)
    osa_rmse = np.sqrt(np.mean((y_osa - Psi[:, sel] @ theta_fr) ** 2)) * sy
    sim_tr_fr = rmse_sim(theta_fr, sel_lbls, un_tr, yn_tr, ny, nu) * sy
    sim_te_fr = rmse_sim(theta_fr, sel_lbls, un_te, yn_te, ny, nu) * sy
    y_sim_tr_fr = simulate(theta_fr, sel_lbls, un_tr, yn_tr, ny, nu) * sy + my
    y_sim_te_fr = simulate(theta_fr, sel_lbls, un_te, yn_te, ny, nu) * sy + my

    semp_lbls, semp_th, _ = semp(Psi, y_osa, lbls, un_tr, yn_tr, ny, nu, n_init=mt)
    sim_tr_se = rmse_sim(semp_th, semp_lbls, un_tr, yn_tr, ny, nu) * sy
    sim_te_se = rmse_sim(semp_th, semp_lbls, un_te, yn_te, ny, nu) * sy
    y_sim_tr_se = simulate(semp_th, semp_lbls, un_tr, yn_tr, ny, nu) * sy + my
    y_sim_te_se = simulate(semp_th, semp_lbls, un_te, yn_te, ny, nu) * sy + my

    summary[name] = {
        'u_tr': u_tr0, 'y_tr': y_tr0, 'u_te': u_te0, 'y_te': y_te0,
        'Ts': Ts, 'ml': ml, 'sy': sy, 'my': my,
        'FROLS': {
            'labels': sel_lbls, 'ERR': ERR, 'n_terms': len(sel),
            'y_sim_tr': y_sim_tr_fr, 'y_sim_te': y_sim_te_fr,
            'osa_rmse': osa_rmse, 'sim_rmse_tr': sim_tr_fr, 'sim_rmse_te': sim_te_fr,
        },
        'SEMP': {
            'labels': semp_lbls, 'n_terms': len(semp_lbls),
            'y_sim_tr': y_sim_tr_se, 'y_sim_te': y_sim_te_se,
            'sim_rmse_tr': sim_tr_se, 'sim_rmse_te': sim_te_se,
        },
    }


# --- Geração de Figuras ---

NAMES = list(loaders.keys())
COLORS = {'measured': '#2C3E50', 'FROLS': '#E74C3C', 'SEMP': '#2980B9'}


def clip_sim(ys, y_ref):
    s = np.std(y_ref)
    return np.clip(ys, y_ref.min() - 5 * s, y_ref.max() + 5 * s)


fig1, axes = plt.subplots(len(NAMES), 2, figsize=(16, 14))
fig1.suptitle('Resultados de Simulação NARX\nFROLS+MGS vs SEMP', fontsize=14, fontweight='bold')

for row, name in enumerate(NAMES):
    d = summary[name]
    ml = d['ml']
    for col, split in enumerate(['tr', 'te']):
        ax = axes[row, col]
        y_true = d['y_tr'] if split == 'tr' else d['y_te']
        y_fr = clip_sim(d['FROLS'][f'y_sim_{split}'], y_true)
        y_se = clip_sim(d['SEMP'][f'y_sim_{split}'], y_true)
        t = np.arange(len(y_true))
        ax.plot(t, y_true, color=COLORS['measured'], lw=1.2, label='Medido', zorder=3)
        ax.plot(t[ml:], y_fr[ml:], color=COLORS['FROLS'], lw=1.0, alpha=0.85, label='FROLS+MGS')
        ax.plot(t[ml:], y_se[ml:], color=COLORS['SEMP'], lw=1.0, alpha=0.85, ls='--', label='SEMP')
        fr_r = d['FROLS'][f'sim_rmse_{split}']
        se_r = d['SEMP'][f'sim_rmse_{split}']
        ax.set_title(f"{name} – {'Treino' if split == 'tr' else 'Teste'}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Amostras', fontsize=8)
        ax.set_ylabel('Saída', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        txt = f"RMSE\nFROLS: {fr_r:.5f}\nSEMP:  {se_r:.5f}"
        ax.text(0.98, 0.05, txt, transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Terceiro Roteiro/images/fig1_sim.png', dpi=140, bbox_inches='tight')
plt.close()

fig2, axes2 = plt.subplots(len(NAMES), 2, figsize=(16, 12))
fig2.suptitle('Erros de Simulação e Espectro ERR', fontsize=13, fontweight='bold')

for row, name in enumerate(NAMES):
    d = summary[name]
    ml = d['ml']
    y_te = d['y_te']
    t = np.arange(len(y_te))
    e_fr = clip_sim(y_te - d['FROLS']['y_sim_te'], y_te)
    e_se = clip_sim(y_te - d['SEMP']['y_sim_te'], y_te)
    ax = axes2[row, 0]
    ax.plot(t[ml:], e_fr[ml:], color=COLORS['FROLS'], lw=0.8, alpha=0.8, label='Erro FROLS+MGS')
    ax.plot(t[ml:], e_se[ml:], color=COLORS['SEMP'], lw=0.8, alpha=0.8, ls='--', label='Erro SEMP')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{name} – Erro (Teste)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Amostras', fontsize=8)
    ax.set_ylabel('y − ŷ', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ERR = d['FROLS']['ERR']
    ax2 = axes2[row, 1]
    clrs = ['#C0392B' if e == ERR.max() else COLORS['FROLS'] for e in ERR]
    ax2.bar(range(len(ERR)), ERR * 100, color=clrs, alpha=0.85, edgecolor='white')
    ax2.set_xticks(range(len(ERR)))
    ax2.set_xticklabels([f"#{i + 1}" for i in range(len(ERR))], fontsize=6)
    ax2.set_xlabel('Posto do termo', fontsize=8)
    ax2.set_ylabel('ERR (%)', fontsize=8)
    ax2.set_title(f'{name} – Espectro ERR (FROLS)', fontsize=9, fontweight='bold')
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.98, 0.95, f'ERR acum. = {np.sum(ERR) * 100:.2f}%',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Terceiro Roteiro/images/fig2_err.png', dpi=140, bbox_inches='tight')
plt.close()

fig3, (ax_bar, ax_tab) = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('Resumo Comparativo: FROLS+MGS vs SEMP', fontsize=13, fontweight='bold')

names_s = ['Ball &\nBeam', 'Heat\nExch.', 'Silverbox', 'Wiener-\nHammer.']
fr_tr = [min(summary[n]['FROLS']['sim_rmse_tr'], 5) for n in NAMES]
se_tr = [min(summary[n]['SEMP']['sim_rmse_tr'], 5) for n in NAMES]
fr_te = [min(summary[n]['FROLS']['sim_rmse_te'], 5) for n in NAMES]
se_te = [min(summary[n]['SEMP']['sim_rmse_te'], 5) for n in NAMES]
x = np.arange(len(NAMES))
w = 0.2
ax_bar.bar(x - 1.5 * w, fr_tr, w, label='FROLS Treino', color=COLORS['FROLS'], alpha=0.7)
ax_bar.bar(x - 0.5 * w, se_tr, w, label='SEMP Treino', color=COLORS['SEMP'], alpha=0.7)
ax_bar.bar(x + 0.5 * w, fr_te, w, label='FROLS Teste', color=COLORS['FROLS'], alpha=1.0, hatch='//')
ax_bar.bar(x + 1.5 * w, se_te, w, label='SEMP Teste', color=COLORS['SEMP'], alpha=1.0, hatch='//')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(names_s, fontsize=9)
ax_bar.set_ylabel('RMSE (limitado a 5)', fontsize=9)
ax_bar.set_title('RMSE de Simulação por Conjunto', fontsize=10, fontweight='bold')
ax_bar.legend(fontsize=8)
ax_bar.grid(True, alpha=0.3, axis='y')

ax_tab.axis('off')
col_h = ['Dataset', 'Método', 'n', 'OSA RMSE', 'Sim RMSE\n(treino)', 'Sim RMSE\n(teste)']
rows_t = []
def fmt(v): return f"{v:.5f}" if v < 1000 else "instável"
for name in NAMES:
    fr = summary[name]['FROLS']
    se = summary[name]['SEMP']
    rows_t.append([name[:12], 'FROLS+MGS', fr['n_terms'],
                   f"{fr['osa_rmse']:.5f}", fmt(fr['sim_rmse_tr']), fmt(fr['sim_rmse_te'])])
    rows_t.append(['', 'SEMP', se['n_terms'], '—', fmt(se['sim_rmse_tr']), fmt(se['sim_rmse_te'])])

tbl = ax_tab.table(cellText=rows_t, colLabels=col_h, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.7)
for j in range(len(col_h)):
    tbl[0, j].set_facecolor('#2C3E50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(rows_t)):
    fc = '#EBF5FB' if (i // 2) % 2 == 0 else 'white'
    for j in range(len(col_h)):
        tbl[i + 1, j].set_facecolor(fc)

ax_tab.set_title('Tabela de Resultados', fontsize=10, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('Terceiro Roteiro/images/fig3_summary.png', dpi=140, bbox_inches='tight')
plt.close()

fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
fig4.suptitle('Estrutura do Modelo NARX – Termos Selecionados\n(azul = retidos pelo SEMP)',
              fontsize=12, fontweight='bold')

for idx, name in enumerate(NAMES):
    ax = axes4[idx // 2, idx % 2]
    d = summary[name]
    fr_lbls = d['FROLS']['labels']
    fr_ERR = d['FROLS']['ERR']
    se_set = set(d['SEMP']['labels'])
    y_pos = np.arange(len(fr_lbls))
    ax.barh(y_pos, fr_ERR * 100, color=COLORS['FROLS'], alpha=0.7, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fr_lbls, fontsize=8)
    ax.set_xlabel('ERR (%)', fontsize=8)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    for i, lbl in enumerate(fr_lbls):
        if lbl in se_set:
            ax.get_yticklabels()[i].set_color(COLORS['SEMP'])
            ax.get_yticklabels()[i].set_fontweight('bold')

plt.tight_layout()
plt.savefig('Terceiro Roteiro/images/fig4_terms.png', dpi=140, bbox_inches='tight')
plt.close()

print("Concluído.")
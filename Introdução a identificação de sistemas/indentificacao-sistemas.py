"""
Identificação de Sistemas - Métodos Determinísticos
=====================================================
Métodos: Ziegler-Nichols, Hägglund, Smith (1ª e 2ª ordem),
         Sundaresan-Krishnaswamy, Mollenkamp

Para cada conjunto de dados, identifica os parâmetros do modelo
FOPDT (1ª ordem) ou SOPDT (2ª ordem) e avalia usando MSE, IAE, ISE, ITAE.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def carregar_dados(caminho):
    """Carrega arquivo com duas colunas: y(t) e t."""
    dados = np.loadtxt(caminho, delimiter=",")
    # detecta qual coluna é o tempo (menor variância relativa / monotônico)
    col0, col1 = dados[:, 0], dados[:, 1]
    if np.all(np.diff(col1) >= 0):   # col1 é monotônica → é o tempo
        t, y = col1, col0
    else:
        t, y = col0, col1
    # garante que t começa em 0
    t = t - t[0]
    return t, y


def normalizar(y):
    """Retorna y normalizado entre 0 e 1 e o ganho K."""
    y_min = y[0]
    y_inf = np.mean(y[int(0.85 * len(y)):])   # média do último 15%
    K = y_inf - y_min
    yn = (y - y_min) / K
    return yn, K, y_min


def interpolar_tempo(t, yn, fracao):
    """Retorna o instante em que yn atinge 'fracao' por interpolação linear."""
    idx = np.searchsorted(yn, fracao)
    if idx == 0:
        return t[0]
    if idx >= len(t):
        return t[-1]
    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = yn[idx - 1], yn[idx]
    if y1 == y0:
        return t0
    return t0 + (fracao - y0) / (y1 - y0) * (t1 - t0)


# SIMULAÇÃO


def simular_fopdt(t, K, tau, L):
    """Resposta ao degrau de G(s) = K·e^(-Ls) / (tau·s + 1)."""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        td = ti - L
        if td > 0:
            y[i] = K * (1 - np.exp(-td / tau))
    return y


def simular_sopdt(t, K, tau1, tau2, L):
    """
    Resposta ao degrau de G(s) = K·e^(-Ls) / ((tau1·s+1)(tau2·s+1)).
    Caso tau1 == tau2 (polo repetido) tratado separadamente.
    """
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        td = ti - L
        if td <= 0:
            continue
        if abs(tau1 - tau2) < 1e-9 * max(tau1, tau2, 1e-12):
            # polo repetido
            y[i] = K * (1 - (1 + td / tau1) * np.exp(-td / tau1))
        else:
            y[i] = K * (1
                        - (tau1 / (tau1 - tau2)) * np.exp(-td / tau1)
                        + (tau2 / (tau1 - tau2)) * np.exp(-td / tau2))
    return y



# CRITÉRIOS DE AVALIAÇÃO

def avaliar(t, y_real, y_modelo):
    e = y_real - y_modelo
    dt = np.diff(t, prepend=t[0])
    N = len(e)
    mse  = np.mean(e**2)
    iae  = np.sum(np.abs(e) * dt)
    ise  = np.sum(e**2 * dt)
    itae = np.sum(t * np.abs(e) * dt)
    return {"MSE": mse, "IAE": iae, "ISE": ise, "ITAE": itae}



# MÉTODOS DE IDENTIFICAÇÃO

def ziegler_nichols(t, yn, K, y_min):
    """
    Método da tangente no ponto de inflexão.
    Retorna (K, tau, L) do modelo FOPDT.
    """
    # derivada suavizada
    dy = np.gradient(yn, t)
    # suavização por média móvel
    w = max(3, len(t) // 30)
    kernel = np.ones(w) / w
    dy_s = np.convolve(dy, kernel, mode="same")

    idx_inf = np.argmax(dy_s)
    t_inf = t[idx_inf]
    y_inf = yn[idx_inf]
    slope = dy_s[idx_inf]

    if slope <= 0:
        # fallback: usa Hägglund
        return hagglund(t, yn, K, y_min)

    # tangente: y - y_inf = slope*(t - t_inf)
    # interseção com y=0 → L
    t_L = t_inf - y_inf / slope
    # interseção com y=1 → L + tau
    t_tau = t_inf + (1 - y_inf) / slope
    L = max(t_L, 0)
    tau = max(t_tau - t_L, 1e-6)
    return {"K": K, "tau": tau, "L": L,
            "modelo": "FOPDT",
            "params_sim": (K, tau, L),
            "sim_fn": simular_fopdt}


def hagglund(t, yn, K, y_min):
    """
    Método de Hägglund (28% e 63%).
    """
    t1 = interpolar_tempo(t, yn, 0.28)
    t2 = interpolar_tempo(t, yn, 0.63)
    L   = max(1.5 * t1 - 0.5 * t2, 0)
    tau = max(t2 - t1, 1e-6)
    return {"K": K, "tau": tau, "L": L,
            "modelo": "FOPDT",
            "params_sim": (K, tau, L),
            "sim_fn": simular_fopdt}


def smith_1ordem(t, yn, K, y_min):
    """
    Método de Smith - 1ª ordem (28.3% e 63.2%).
    """
    t1 = interpolar_tempo(t, yn, 0.283)
    t2 = interpolar_tempo(t, yn, 0.632)
    tau = max(2 / 3 * (t2 - t1), 1e-6)
    L   = max(t2 - tau, 0)
    return {"K": K, "tau": tau, "L": L,
            "modelo": "FOPDT",
            "params_sim": (K, tau, L),
            "sim_fn": simular_fopdt}


def smith_2ordem(t, yn, K, y_min):
    """
    Método de Smith - 2ª ordem.
    Usa t20 (20%) e t60 (60%) para determinar razão e depois
    estimar tau1, tau2, L via tabela aproximada por fórmulas.
    Referência: Smith (1985), adaptado.
    """
    t1 = interpolar_tempo(t, yn, 0.283)
    t2 = interpolar_tempo(t, yn, 0.632)

    # razão característica de Smith
    r = t1 / t2 if t2 > 0 else 0.5

    # estimativa de L e tau_total via Smith 1ª ordem como base
    tau_total = max(2 / 3 * (t2 - t1), 1e-6)
    L = max(t2 - tau_total, 0)

    # para 2ª ordem: distribui tau_total em dois polos
    # usando a relação de Smith: razão r determina distribuição
    # (tabela de Smith aproximada por polinômio)
    if r <= 0.32:
        alpha = 0.1
    elif r <= 0.46:
        alpha = 0.3
    elif r <= 0.55:
        alpha = 0.5
    else:
        alpha = 0.7

    tau1 = max(tau_total * (1 - alpha), 1e-6)
    tau2 = max(tau_total * alpha, 1e-6)

    return {"K": K, "tau1": tau1, "tau2": tau2, "L": L,
            "modelo": "SOPDT",
            "params_sim": (K, tau1, tau2, L),
            "sim_fn": simular_sopdt}


def sundaresan_krishnaswamy(t, yn, K, y_min):
    """
    Método de Sundaresan & Krishnaswamy (35.3% e 85.3%).
    """
    t1 = interpolar_tempo(t, yn, 0.353)
    t2 = interpolar_tempo(t, yn, 0.853)
    tau = max(0.67 * (t2 - t1), 1e-6)
    L   = max(1.3 * t1 - 0.29 * t2, 0)
    return {"K": K, "tau": tau, "L": L,
            "modelo": "FOPDT",
            "params_sim": (K, tau, L),
            "sim_fn": simular_fopdt}


def mollenkamp(t, yn, K, y_min):
    """
    Método de Mollenkamp (25%, 50%, 75%).
    Estima modelo SOPDT resolvendo sistema implícito.
    Referência: Mollenkamp (1965), adaptado de Coelho & Coelho (2004).
    """
    t1 = interpolar_tempo(t, yn, 0.25)
    t2 = interpolar_tempo(t, yn, 0.50)
    t3 = interpolar_tempo(t, yn, 0.75)

    # razões
    if t3 <= 0:
        return smith_1ordem(t, yn, K, y_min)

    r1 = t1 / t3
    r2 = t2 / t3

    # Estimativa de L via aproximação de Mollenkamp
    L = max(t1 * (t1 / t2 - 1) / (1 - t1 / t2 * np.exp(1 - t1/t2)) 
            if t2 > 0 else 0, 0)

    # Estimativa de tau_total a partir de t3
    # Para SOPDT: t3 corresponde aproximadamente a ~(tau1+tau2+L)*1.5
    tau_total = max((t3 - L) * 0.75, 1e-6)

    # Distribuição dos polos baseada na razão r = t1/t2
    r = t1 / t2 if t2 > 0 else 0.5
    # r próximo de 0.5 → polos iguais; r menor → mais separados
    alpha = max(min(1 - 2 * r, 0.8), 0.05)
    tau1 = max(tau_total * (1 + alpha) / 2, 1e-6)
    tau2 = max(tau_total * (1 - alpha) / 2, 1e-6)

    return {"K": K, "tau1": tau1, "tau2": tau2, "L": L,
            "modelo": "SOPDT",
            "params_sim": (K, tau1, tau2, L),
            "sim_fn": simular_sopdt}



METODOS = {
    "Ziegler-Nichols":       ziegler_nichols,
    "Hägglund":              hagglund,
    "Smith 1ª ordem":        smith_1ordem,
    "Smith 2ª ordem":        smith_2ordem,
    "Sundaresan-Krishnaswamy": sundaresan_krishnaswamy,
    "Mollenkamp":            mollenkamp,
}


def processar_conjunto(caminho):
    t, y = carregar_dados(caminho)
    yn, K, y_min = normalizar(y)
    resultados = {}
    for nome, metodo in METODOS.items():
        try:
            res = metodo(t, yn, K, y_min)
            # simula resposta do modelo e desfaz normalização
            y_sim_n = res["sim_fn"](t, *res["params_sim"])
            y_sim = y_sim_n + y_min          # volta à escala original
            crit = avaliar(t, y, y_sim)
            resultados[nome] = {**res, "criterios": crit, "y_sim": y_sim}
        except Exception as e:
            resultados[nome] = {"erro": str(e)}
    return t, y, yn, K, y_min, resultados


def imprimir_tabela(nome_arq, resultados):
    print(f"\n{'='*70}")
    print(f"  {nome_arq}")
    print(f"{'='*70}")
    header = f"{'Método':<26} {'Modelo':<8} {'MSE':>10} {'IAE':>10} {'ISE':>10} {'ITAE':>10}"
    print(header)
    print("-" * 70)
    for nome, res in resultados.items():
        if "erro" in res:
            print(f"  {nome:<24}  ERRO: {res['erro']}")
            continue
        c = res["criterios"]
        modelo = res.get("modelo", "?")
        print(f"  {nome:<24}  {modelo:<8} "
              f"{c['MSE']:>10.5f} {c['IAE']:>10.5f} "
              f"{c['ISE']:>10.5f} {c['ITAE']:>10.5f}")

    # melhor método por ITAE
    validos = {k: v for k, v in resultados.items() if "criterios" in v}
    if validos:
        melhor = min(validos, key=lambda k: validos[k]["criterios"]["ITAE"])
        print(f"\n  → Melhor por ITAE: {melhor}")


def plotar_resultados(nome_arq, t, y, resultados, salvar=True):
    n_met = len(METODOS)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Identificação de Sistemas — {nome_arq}", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    cores = plt.cm.tab10.colors
    for i, (nome, res) in enumerate(resultados.items()):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(t, y, "k.", markersize=2, alpha=0.5, label="Dados reais")
        if "y_sim" in res:
            ax.plot(t, res["y_sim"], color=cores[i], linewidth=1.8, label="Modelo")
            c = res["criterios"]
            ax.set_title(f"{nome}\nMSE={c['MSE']:.4f}  IAE={c['IAE']:.4f}", fontsize=8)
        else:
            ax.set_title(f"{nome}\nERRO", fontsize=8)
        ax.set_xlabel("Tempo", fontsize=7)
        ax.set_ylabel("Saída", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    nome_saida = Path(nome_arq).stem + "_resultado.png"
    if salvar:
        fig.savefig(nome_saida, dpi=120, bbox_inches="tight")
        print(f"  Gráfico salvo: {nome_saida}")
    plt.close(fig)



if __name__ == "__main__":
    pasta = Path(".")
    arquivos = sorted(pasta.glob("conjunto*.txt"))

    if not arquivos:
        print("Nenhum arquivo conjunto*.txt encontrado na pasta atual.")
        print("Coloque os arquivos conjunto1.txt ... conjunto6.txt aqui e rode novamente.")
    else:
        print(f"Encontrados {len(arquivos)} arquivo(s): {[a.name for a in arquivos]}")

        for arq in arquivos:
            t, y, yn, K, y_min, resultados = processar_conjunto(arq)
            imprimir_tabela(arq.name, resultados)
            plotar_resultados(arq.name, t, y, resultados, salvar=True)

        print("\n\nProcessamento concluído.")
        print("Para cada conjunto foi gerado um arquivo _resultado.png com os gráficos.")
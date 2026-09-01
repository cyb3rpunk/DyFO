"""Script to inject the Interactive Quantitative Finance & AI Glossary Modal into bracis_presentation_deck.html."""

from pathlib import Path

deck_path = Path("doc/bracis_presentation_deck.html")

with open(deck_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS for Glossary Button & Modal
css_to_add = """
    /* Glossary Button */
    .glossary-btn {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(168, 85, 247, 0.15));
      color: var(--accent-blue);
      border: 1px solid rgba(56, 189, 248, 0.4);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }

    .glossary-btn:hover {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.3), rgba(168, 85, 247, 0.3));
      border-color: var(--accent-blue);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }

    /* Glossary Modal */
    .glossary-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(11, 19, 41, 0.82);
      backdrop-filter: blur(12px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 30px;
    }

    .glossary-modal.open {
      display: flex;
      animation: modalFadeIn 0.25s ease-out;
    }

    @keyframes modalFadeIn {
      from { opacity: 0; transform: scale(0.97); }
      to { opacity: 1; transform: scale(1); }
    }

    .glossary-box {
      background: #152238;
      border: 1px solid #38bdf8;
      border-radius: 16px;
      width: 90%;
      max-width: 1100px;
      height: 86vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 25px rgba(56, 189, 248, 0.2);
      overflow: hidden;
    }

    .glossary-header {
      padding: 18px 24px;
      background: rgba(21, 34, 56, 0.95);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .glossary-title {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .glossary-close {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1.5rem;
      cursor: pointer;
      line-height: 1;
      transition: color 0.2s;
    }

    .glossary-close:hover {
      color: var(--accent-red);
    }

    .glossary-search-bar {
      padding: 12px 24px;
      background: #0b1329;
      border-bottom: 1px solid var(--border-color);
    }

    .glossary-search-input {
      width: 100%;
      background: #152238;
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 0.92rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .glossary-search-input:focus {
      border-color: var(--accent-blue);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }

    .glossary-content {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
      gap: 18px;
    }

    .glossary-card {
      background: #0b1329;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.2s, transform 0.2s;
    }

    .glossary-card:hover {
      border-color: var(--accent-blue);
      transform: translateY(-2px);
    }

    .glossary-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
    }

    .glossary-term {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--accent-emerald);
    }

    .glossary-ai-badge {
      background: rgba(168, 85, 247, 0.2);
      color: var(--accent-purple);
      border: 1px solid rgba(168, 85, 247, 0.4);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      white-space: nowrap;
    }

    .glossary-def {
      font-size: 0.88rem;
      color: var(--text-main);
      line-height: 1.45;
    }

    .glossary-pitch {
      background: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent-blue);
      padding: 6px 12px;
      border-radius: 0 6px 6px 0;
      font-size: 0.82rem;
      color: #bae6fd;
      font-style: italic;
    }
"""

if ".glossary-btn" not in html:
    html = html.replace("/* Header Controls */", css_to_add + "\n    /* Header Controls */")
    print("Inserted Glossary CSS.")

# 2. Add Glossary Button to Header
old_header_right = """      <div class="header-right">
        <select class="slide-select" id="slideSelect" onchange="showSlide(parseInt(this.value))">"""

new_header_right = """      <div class="header-right">
        <button class="glossary-btn" onclick="toggleGlossary()" title="Atalho: Tecla [G]">📚 Glossário IA ↔ Finanças</button>
        <select class="slide-select" id="slideSelect" onchange="showSlide(parseInt(this.value))">"""

if old_header_right in html:
    html = html.replace(old_header_right, new_header_right)
    print("Inserted Glossary Button in Header.")

# 3. Add Modal HTML before </body>
modal_html = """
  <!-- Interactive Quantitative Finance & AI Glossary Modal -->
  <div class="glossary-modal" id="glossaryModal" onclick="handleModalBackdropClick(event)">
    <div class="glossary-box">
      <div class="glossary-header">
        <div class="glossary-title">
          <span>📚 Glossário &amp; Ponte Interdisciplinar: Finanças Quantitativas ↔ IA</span>
          <span style="font-size:0.75rem; background:rgba(56,189,248,0.2); color:#38bdf8; padding:2px 8px; border-radius:4px;">Atalho: [G] ou [Esc]</span>
        </div>
        <button class="glossary-close" onclick="toggleGlossary()">&times;</button>
      </div>
      <div class="glossary-search-bar">
        <input type="text" class="glossary-search-input" id="glossarySearch" placeholder="🔍 Filtrar conceitos (ex: Covariância, SPD, 1/N, Turnover, Sharpe, Higham, HRP)..." onkeyup="filterGlossary()">
      </div>
      <div class="glossary-content" id="glossaryCardsContainer">
        
        <div class="glossary-card" data-keywords="covariancia spd higham matriz adjacencia energia cone">
          <div class="glossary-card-header">
            <span class="glossary-term">1. Matriz de Covariância &amp; Cone SPD</span>
            <span class="glossary-ai-badge">Matriz de Adjacência Causal</span>
          </div>
          <p class="glossary-def">Representa o acoplamento causal entre os nós do grafo. Pertence ao cone de matrizes simétricas estritamente positivas definidas (\(\\mathcal{S}_{++}^N\)). Se um autovalor for negativo (\(\\lambda_{\\min} \\le 0\)), otimizadores convexos divergem com variância negativa.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;A covariância é a matriz de adjacência ponderada do grafo financeiro. O DyFO aplica a projeção de Higham para garantir que a energia do sistema nunca seja negativa.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="equal weight 1/n demiguel benchmark entropia uniforme">
          <div class="glossary-card-header">
            <span class="glossary-term">2. Benchmark Equal-Weight (1/N)</span>
            <span class="glossary-ai-badge">Distribuição Uniforme (Máx. Entropia)</span>
          </div>
          <p class="glossary-def">Alocação uniforme (\(w_i = 1/N\)). É o benchmark mais difícil da literatura (DeMiguel et al., 2009) por ter erro zero de estimação de parâmetros e turnover nulo. O DyFO-Tangency o supera com sinais topológicos de momento e centralidade.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;O 1/N é o baseline de máxima entropia. Modelos de IA costumam perder para ele pelo excesso de giro. O DyFO o supera combinando centralidade de grafo com controle estrito de custos.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="gmvp minima variancia markowitz otimizacao quadratica simplex">
          <div class="glossary-card-header">
            <span class="glossary-term">3. Global Minimum Variance (GMVP)</span>
            <span class="glossary-ai-badge">Otimização Quadrática Convexa no Simplex</span>
          </div>
          <p class="glossary-def">Busca a carteira de menor variância (\(\\min \\frac{1}{2} \\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}\) s.t. \(\\sum w_i=1, w_i \\ge 0\)). Não tenta adivinhar retornos futuros, isolando perfeitamente a qualidade e estabilidade da matriz de covariância prevista pelo DyFO.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;O GMVP é o teste de Turing da matriz de covariância: ele minimiza o ruído do sistema sem ruído de predição de preços.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="turnover price drift custos transacao 10 bps atrito">
          <div class="glossary-card-header">
            <span class="glossary-term">4. Price Drift &amp; Turnover Diário (10 bps)</span>
            <span class="glossary-ai-badge">Atrito L1 de Transição de Estado</span>
          </div>
          <p class="glossary-def">O drift (\(w_{t-1}^+\)) é a flutuação passiva dos pesos causada pelo mercado. O turnover (\(\\|\\mathbf{w}_t - \\mathbf{w}_{t-1}^+\\|_1\)) mede o volume negociado. Cada 100% de giro custa 10 bps (0.10%), penalizando modelos que mudam de opinião com excesso de frequência.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;Mudar de ideia todo dia tem custo financeiro real. Nosso benchmark desconta 10 bps a cada passo, provando que a suavização do DyFO economiza mais de 500 bps/ano.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="sharpe ratio dsr moody saffell relacao sinal ruido snr">
          <div class="glossary-card-header">
            <span class="glossary-term">5. Sharpe Líquido &amp; Differential Sharpe (DSR)</span>
            <span class="glossary-ai-badge">SNR Regularizada por Custo</span>
          </div>
          <p class="glossary-def">Mede a relação sinal-ruído (\(R_{\\text{net}} / \\sigma_{\\text{net}}\)). Em DRL, usamos a formulação recursiva online de Moody &amp; Saffell (2001) para guiar o gradiente da política PPO diretamente na direção do máximo retorno ajustado ao risco.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;O DSR é o gradiente passo-a-passo da eficiência de risco, punindo variância e giro de carteira a cada ação do agente de IA.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="espectral autovalores trace contagio panico cash buffer">
          <div class="glossary-card-header">
            <span class="glossary-term">6. Concentração Espectral (\(\\lambda_1 / \\text{Tr}(\\mathbf{\\Sigma})\))</span>
            <span class="glossary-ai-badge">Colapso de Dimensionalidade do Grafo</span>
          </div>
          <p class="glossary-def">Quando o primeiro autovalor domina mais de 38% do traço da matriz, os nós tornam-se colineares (contágio sistêmico / pânico). O DyFO detecta esse colapso e aloca preventivamente em Caixa Livre de Risco (\(\\ge 30\\%\)).</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;Quando o grafo perde diversidade dimensional, o DyFO identifica o contágio antes do crash e estanca as perdas alocando em caixa.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="hrp lopez de prado clusterizacao arvore dendrograma quase diagonalizacao">
          <div class="glossary-card-header">
            <span class="glossary-term">7. Hierarchical Risk Parity (GraphHRP)</span>
            <span class="glossary-ai-badge">Clusterização em Árvore &amp; Bisseção Recursiva</span>
          </div>
          <p class="glossary-def">Agrupa ativos em dendrograma sobre a distância de correlação DyFO (\(d_{ij} = \\sqrt{(1 - \\rho_{ij})/2}\)) e particiona variâncias recursivamente sem inversão de matrizes (\(\\mathbf{\\Sigma}^{-1}\)), eliminando instabilidades numéricas.</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;O GraphHRP substitui a inversão frágil de matrizes por uma árvore hierárquica de divisão e conquista guiada pela topologia do DyFO.&rdquo;</div>
        </div>

        <div class="glossary-card" data-keywords="volatility targeting meta volatilidade invariancia normalizacao">
          <div class="glossary-card-header">
            <span class="glossary-term">8. Volatility Targeting (Meta-Volatilidade)</span>
            <span class="glossary-ai-badge">Normalização Invariante de Ação</span>
          </div>
          <p class="glossary-def">Escala dinamicamente a exposição da carteira (\(k_t = \\sigma^* / \\hat{\\sigma}_{p, t}\)) para manter a volatilidade estável em 12% ao ano, acelerando em mercados calmos e desalavancando em mercados voláteis (+339 bps sobre o 1/N).</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;Ajustamos dinamicamente o volume de apostas pelo inverso da incerteza prevista pelo DyFO, maximizando a riqueza acumulada líquida.&rdquo;</div>
        </div>

      </div>
    </div>
  </div>
"""

if "id=\"glossaryModal\"" not in html:
    html = html.replace("</body>", modal_html + "\n</body>")
    print("Inserted Glossary Modal HTML.")

# 4. Add JavaScript logic for modal and search
js_to_add = """
    /* Glossary Functions */
    function toggleGlossary() {
      const modal = document.getElementById('glossaryModal');
      if (!modal) return;
      modal.classList.toggle('open');
      if (modal.classList.contains('open')) {
        const input = document.getElementById('glossarySearch');
        if (input) {
          input.value = '';
          filterGlossary();
          setTimeout(() => input.focus(), 50);
        }
      }
    }

    function handleModalBackdropClick(event) {
      if (event.target.id === 'glossaryModal') {
        toggleGlossary();
      }
    }

    function filterGlossary() {
      const query = document.getElementById('glossarySearch').value.toLowerCase().trim();
      const cards = document.querySelectorAll('.glossary-card');
      cards.forEach(card => {
        const text = (card.innerText + ' ' + (card.getAttribute('data-keywords') || '')).toLowerCase();
        if (!query || text.includes(query)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    }

    // Keyboard shortcut [G] and [Escape]
    document.addEventListener('keydown', (e) => {
      if (e.key === 'g' || e.key === 'G') {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
          e.preventDefault();
          toggleGlossary();
        }
      } else if (e.key === 'Escape') {
        const modal = document.getElementById('glossaryModal');
        if (modal && modal.classList.contains('open')) {
          toggleGlossary();
        }
      }
    });
"""

if "toggleGlossary" not in html:
    html = html.replace("</script>\n</body>", js_to_add + "\n  </script>\n</body>")
    print("Inserted Glossary JS logic.")

with open(deck_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Successfully injected Interactive Glossary Modal into bracis_presentation_deck.html!")

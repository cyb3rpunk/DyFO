"""Script to inject Slides 12 and 13 (AI Frontiers & Beating Equal-Weight) into bracis_presentation_deck.html."""

import base64
from pathlib import Path

deck_path = Path("doc/bracis_presentation_deck.html")
fig_beat_ew = Path("figures/demo_dyfo_beat_equal_weight.png")
fig_frontiers = Path("figures/demo_dyfo_llm_neurosymbolic.png")

with open(fig_beat_ew, "rb") as f:
    img_beat_ew_b64 = base64.b64encode(f.read()).decode("utf-8")

with open(fig_frontiers, "rb") as f:
    img_frontiers_b64 = base64.b64encode(f.read()).decode("utf-8")

with open(deck_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update Header Select Dropdown (1 to 15)
old_select = """            <select class="slide-select" id="slideSelect" onchange="showSlide(parseInt(this.value))">
        <option value="1">1. Título & Visão Geral da Pesquisa</option>
        <option value="2">2. Fundamentos de IA: De GNNs Estáticas a TGAT</option>
        <option value="3">3. A Proposta do DyFO & Redes Heterogêneas</option>
        <option value="4">4. Diagnóstico: Diluição de Atenção no TGAT</option>
        <option value="5">5. Relation-Aware TGAT v2 (DyFO)</option>
        <option value="6">6. Escalaridade Combinatória (N=18, 50, 100)</option>
        <option value="7">7. Validação Empírica & Benchmarks SOTA</option>
        <option value="8">8. Estudo de Ablação: De Diluição a Sinergia</option>
        <option value="9">9. Regularização em DRL & Quebra de Simetria</option>
        <option value="10">10. Robustez em Regimes de Estresse (COVID-19)</option>
        <option value="11">11. Aplicação Prática & Utilidade no Mundo Real</option>
        <option value="12">12. Tríade do Doutorado & Ontologia Semântica</option>
        <option value="13">13. Conclusões & Contribuições em IA no BRACIS</option>
      </select>
      <div class="slide-counter" id="slideIndicator">Slide 1 de 13</div>"""

new_select = """            <select class="slide-select" id="slideSelect" onchange="showSlide(parseInt(this.value))">
        <option value="1">1. Título & Visão Geral da Pesquisa</option>
        <option value="2">2. Fundamentos de IA: De GNNs Estáticas a TGAT</option>
        <option value="3">3. A Proposta do DyFO & Redes Heterogêneas</option>
        <option value="4">4. Diagnóstico: Diluição de Atenção no TGAT</option>
        <option value="5">5. Relation-Aware TGAT v2 (DyFO)</option>
        <option value="6">6. Escalaridade Combinatória (N=18, 50, 100)</option>
        <option value="7">7. Validação Empírica & Benchmarks SOTA</option>
        <option value="8">8. Estudo de Ablação: De Diluição a Sinergia</option>
        <option value="9">9. Regularização em DRL & Quebra de Simetria</option>
        <option value="10">10. Robustez em Regimes de Estresse (COVID-19)</option>
        <option value="11">11. Aplicação Prática & Utilidade no Mundo Real</option>
        <option value="12">12. Fronteiras de IA: Neuro-Simbólico, DRL & DQN</option>
        <option value="13">13. Superando o Benchmark 1/N & Atribuição de Custo</option>
        <option value="14">14. Tríade do Doutorado & Ontologia Semântica</option>
        <option value="15">15. Conclusões & Contribuições em IA no BRACIS</option>
      </select>
      <div class="slide-counter" id="slideIndicator">Slide 1 de 15</div>"""

if old_select in html:
    html = html.replace(old_select, new_select)
    print("Updated select dropdown to 15 slides.")
else:
    print("Warning: old_select not found verbatim.")

# 2. Construct Slides 12 and 13 HTML
slides_12_13_html = f"""    <!-- SLIDE 12: Advanced AI Frontiers (Neuro-Symbolic, DRL, DQN) -->
    <div class="slide" id="slide-12">
      <div class="slide-header">
        <div class="slide-category">Fronteiras de Inteligência Artificial & Paradigmas Híbridos</div>
        <h2 class="slide-title">Fronteiras de IA no DyFO: Neuro-Simbólico GraphRAG, Continuous DRL & Discrete DQN</h2>
      </div>
      <div class="slide-content">
        <div class="text-column">
          <p><strong>Conexão do Motor Topológico/Covariância do DyFO aos Três Pilares da IA Moderna:</strong></p>
          <ul>
            <li><strong>1. Neuro-Symbolic AI & LLM GraphRAG:</strong> Extração causal de subgrafos locais (ego-networks \(\\Delta\\hat{{\\rho}}\) e eventos macro) serializados em RDF/JSON-LD. O LLM traduz regras semânticas em restrições convexas estritas (\(A_{{\\text{{ub}}}} \\mathbf{{w}} \\le \\mathbf{{b}}_{{\\text{{ub}}}}\)), garantindo conformidade regulatória com 100% de explicabilidade.</li>
            <li><strong>2. Continuous DRL (Relational Actor-Critic PPO):</strong> Estados relacionais \(\\mathbf{{S}}_t = [\\mathbf{{Z}}_t, \\mathbf{{w}}_{{t-1}}, \\boldsymbol{{\\pi}}_t] \\in \\mathbb{{R}}^{{N \\times 105}}\) com atenção cruzada multi-head entre ativos, quebrando a degeneração de simetria e otimizando o Differential Sharpe Ratio (+42 bps líquidos sobre DRL sem grafo).</li>
            <li><strong>3. Discrete DQN (Regime-Switching Hedging):</strong> Estados espectrais (top-5 autovalores \(\\lambda_k\), spectral gap, centralidade). O agente comuta dinamicamente entre Alpha GMVP, Defensive ERC e Tail-Risk Cash Buffer, evitando \(3113\\text{{ bps}}\) de prejuízo transacional em relação a modelos não-suavizados.</li>
          </ul>
          <div class="callout">
            <strong>Sinergia Multimodal:</strong> O DyFO atua como a âncora estrutural de baixa latência e causalidade matemática que alimenta tanto modelos simbólicos (LLMs) quanto conexionistas (DRL/DQN).
          </div>
        </div>
        <div class="img-column">
          <img src="data:image/png;base64,{img_frontiers_b64}" alt="Fronteiras de IA DyFO: Neuro-Symbolic GraphRAG">
          <div class="caption">Pipeline Neuro-Simbólico DyFO: Extração de Grafo Causal (A), Síntese de Restrições via LLM (B), Solver Quadrático com Projeção SPD (C) e Explicabilidade Semântica Auditável (D).</div>
        </div>
      </div>
    </div>

    <!-- SLIDE 13: Beating Equal-Weight (1/N) & Cost Attribution -->
    <div class="slide" id="slide-13">
      <div class="slide-header">
        <div class="slide-category">Econometria & Microestrutura de Mercado</div>
        <h2 class="slide-title">Superando o Benchmark 1/N sob Custos Reais & Análise de Atribuição</h2>
      </div>
      <div class="slide-content">
        <div class="text-column">
          <p><strong>Resolvendo o Paradoxo de DeMiguel et al. (2009) com Controle Estrito de Custos (10 bps):</strong></p>
          <ul>
            <li><strong>Superação do Equal-Weight (1/N):</strong> O modelo <em>DyFO-Tangency</em> (Momentum 20d + Centralidade do DyFO) superou o 1/N em Retorno Líquido (<span class="highlight-green">15.80% vs 15.71%</span>), Sharpe Líquido (<span class="highlight-green">0.9927 vs 0.9868</span>) e Max Drawdown (<span class="highlight-green">-12.26% vs -12.36%</span>).</li>
            <li><strong>Domínio com Meta-Volatilidade (VolTarget 12%):</strong> Entregou <span class="highlight-green">19.10% de Retorno Líquido (+339 bps sobre o 1/N)</span> e Riqueza Líquida de <span class="highlight-green">1.1851x</span> com proteção dinâmica de caixa em choques espectrais.</li>
            <li><strong>Atribuição de Benefício vs Prejuízo:</strong>
              <br>&bull; <em>Benefício de Filtragem Topológica:</em> Redução de 83.4% no turnover do EWMA, economizando <strong>589.2 bps</strong> em custos.
              <br>&bull; <em>Benefício em DRL:</em> Embeddings relacionais geram <strong>+73 bps de retorno bruto</strong> e <strong>+42 bps líquidos</strong>.
              <br>&bull; <em>Trade-off de Restrições LLM:</em> Custo de -164 bps em troca de governança e explicabilidade auditável.
            </li>
          </ul>
          <div class="callout">
            <strong>Protocolo Walk-Forward Contínuo:</strong> Avaliação out-of-sample (250 dias) com drift de preços \(w_{{t-1}}^+\) e dedução diária de 10 bps por turnover.
          </div>
        </div>
        <div class="img-column">
          <img src="data:image/png;base64,{img_beat_ew_b64}" alt="Benchmark Superando Equal-Weight 1/N">
          <div class="caption">Simulação Walk-Forward Out-of-Sample: Riqueza Líquida Acumulada (1), Trajetórias de Drawdown (2), Sharpe Rolling 60d (3) e Giro Diário vs Cost Drag Anual (4).</div>
        </div>
      </div>
    </div>

"""

# 3. Rename old slide 12 to 14, and old slide 13 to 15
old_slide_12_marker = '    <!-- SLIDE 12: Doctoral Triad Integration -->\n    <div class="slide" id="slide-12">'
new_slide_14_marker = '    <!-- SLIDE 14: Doctoral Triad Integration -->\n    <div class="slide" id="slide-14">'

old_slide_13_marker = '    <!-- SLIDE 13: Conclusions & Contributions -->\n    <div class="slide" id="slide-13">'
new_slide_15_marker = '    <!-- SLIDE 15: Conclusions & Contributions -->\n    <div class="slide" id="slide-15">'

if old_slide_12_marker in html:
    html = html.replace(old_slide_12_marker, slides_12_13_html + new_slide_14_marker)
    html = html.replace(old_slide_13_marker, new_slide_15_marker)
    print("Inserted Slides 12 & 13 and renamed Slides 14 & 15.")
else:
    print("Warning: old_slide_12_marker not found.")

# 4. Update JS totalSlides
html = html.replace("const totalSlides = 13;", "const totalSlides = 15;")

with open(deck_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Successfully updated bracis_presentation_deck.html to 15 interactive slides!")

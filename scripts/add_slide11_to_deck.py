"""Script to insert Slide 11 (Practical Portfolio & Risk Utility) into bracis_presentation_deck.html."""

import base64
from pathlib import Path

deck_path = Path("doc/bracis_presentation_deck.html")
img_path = Path("figures/demo_dyfo_practical_portfolio.png")

with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

with open(deck_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update Header Select and Slide Counter
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
        <option value="11">11. Tríade do Doutorado & Ontologia Semântica</option>
        <option value="12">12. Conclusões & Contribuições em IA no BRACIS</option>
      </select>
      <div class="slide-counter" id="slideIndicator">Slide 1 de 12</div>"""

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
        <option value="12">12. Tríade do Doutorado & Ontologia Semântica</option>
        <option value="13">13. Conclusões & Contribuições em IA no BRACIS</option>
      </select>
      <div class="slide-counter" id="slideIndicator">Slide 1 de 13</div>"""

if old_select in html:
    html = html.replace(old_select, new_select)
    print("Updated select dropdown and counter.")
else:
    print("WARNING: old_select not found exactly, check formatting.")

# 2. Construct New Slide 11
slide_11_html = f"""    <!-- SLIDE 11: Practical Utility & Real-World Portfolio Demo -->
    <div class="slide" id="slide-11">
      <div class="slide-header">
        <div class="slide-category">Asset Management & Engenharia de Risco em Produção</div>
        <h2 class="slide-title">Aplicação Prática no Mundo Real: Do Grafo Causal à Gestão de Portfólio</h2>
      </div>
      <div class="slide-content">
        <div class="text-column">
          <p><strong>Operacionalização em Fundos Quantitativos e Sistemas Autônomos de Trading:</strong></p>
          <ul>
            <li><strong>Resolução do "Erro de Markowitz":</strong> Matrizes amostrais empíricas sofrem de instabilidade \(\mathcal{{O}}(N^2)\). O DyFO aplica projeção de Higham (2002) e Shrinkage Híbrido (\(\lambda_{{\\min}} \\ge 10^{{-4}}\)), garantindo semidefinição positiva estrita para GMVP e Equal Risk Contribution (ERC).</li>
            <li><strong>Mitigação de Caudas Gordas e Crash Protection:</strong> A injeção contínua de eventos macro (\(\\texttt{{FED\\_DECISION}}\), \(\\texttt{{EARNINGS}}\)) reduz o Max Drawdown para <span class="highlight-green">-4.51%</span> (menor risco de cauda entre redes neurais e modelos econométricos).</li>
            <li><strong>Controle Estrito de Custos Operacionais:</strong> O mecanismo temporal contínuo amortece o ruído diário espúrio, entregando turnover de <span class="highlight-gold">0.0847</span> (redução de 33% sobre GNNs estáticas), evitando a corrosão de alfa por slippage.</li>
            <li><strong>Quebra de Degeneração Heurística 1/N:</strong> Embeddings relacionais \(\\mathbf{{z}}_i(t) \\in \\mathbb{{R}}^{{100}}\) quebram a simetria de ativos em agentes de Deep RL (+1.72% de Sharpe consistente vs baselines uniformes).</li>
          </ul>
          <div class="callout">
            <strong>Demonstração End-to-End:</strong> Simulação causal diária out-of-sample (1 ano completo, S&amp;P 500 balanceado GICS N=30) gerando retorno anualizado de <span class="highlight-green">26.74%</span> e Sharpe Realizado de <span class="highlight-green">2.31</span> sob restrição GMVP.
          </div>
        </div>
        <div class="img-column">
          <img src="data:image/png;base64,{img_b64}" alt="Demonstração Prática de Portfólio DyFO">
          <div class="caption">Execução out-of-sample (250 dias): Curva de Patrimônio Acumulado (A), Volatilidade Realizada 30d (B), Trajetória de Drawdowns (C) e Alocação Setorial Dinâmica DyFO (D).</div>
        </div>
      </div>
    </div>

"""

# 3. Rename old slide-11 to slide-12 and old slide-12 to slide-13
old_slide_11_marker = '    <!-- SLIDE 11: Doctoral Triad Integration -->\n    <div class="slide" id="slide-11">'
new_slide_12_marker = '    <!-- SLIDE 12: Doctoral Triad Integration -->\n    <div class="slide" id="slide-12">'

old_slide_12_marker = '    <!-- SLIDE 12: Conclusions & Contributions -->\n    <div class="slide" id="slide-12">'
new_slide_13_marker = '    <!-- SLIDE 13: Conclusions & Contributions -->\n    <div class="slide" id="slide-13">'

if old_slide_11_marker in html:
    # Insert slide 11 before old slide 11, and rename old slide 11 and 12
    html = html.replace(old_slide_11_marker, slide_11_html + new_slide_12_marker)
    html = html.replace(old_slide_12_marker, new_slide_13_marker)
    print("Inserted Slide 11 and renamed Slide 12 and Slide 13.")
else:
    print("WARNING: old_slide_11_marker not found, check formatting.")

# 4. Update JS totalSlides
html = html.replace("const totalSlides = 12;", "const totalSlides = 13;")
print("Updated const totalSlides = 13 in script.")

with open(deck_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Deck HTML successfully updated with Slide 11!")

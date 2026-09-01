"""Add TGAT card to Glossary Modal in bracis_presentation_deck.html."""

from pathlib import Path

deck_path = Path("doc/bracis_presentation_deck.html")

with open(deck_path, "r", encoding="utf-8") as f:
    html = f.read()

tgat_card_html = """        <div class="glossary-card" data-keywords="tgat xu bochner atencao temporal diluicao relation aware heterogenea">
          <div class="glossary-card-header">
            <span class="glossary-term">0. TGAT Original vs. Relation-Aware TGAT (DyFO)</span>
            <span class="glossary-ai-badge">Arquitetura Central de IA</span>
          </div>
          <p class="glossary-def">O TGAT original (Xu et al., ICLR 2020) usa codificação harmônica de Bochner em tempo contínuo, mas sofre de <em>diluição homogênea de atenção</em> em grafos densos O(N²). O DyFO cura essa patologia injetando o vetor de tipo de aresta (\(\\mathbf{W}_E \\mathbf{e}_{ij}\)) e desacoplando a agregação intra-relação da fusão semântica inter-relações (\(R^2: 0.852 \\to 0.893\)).</p>
          <div class="glossary-pitch"><strong>Como Explicar:</strong> &ldquo;O TGAT original tenta ouvir 100 vozes juntas em uma festa. O DyFO atua como uma mesa de som que separa e equaliza cada canal (setores, supply chain e macro) antes da mixagem final.&rdquo;</div>
        </div>

"""

target_marker = '<div class="glossary-content" id="glossaryCardsContainer">\n'

if "0. TGAT Original vs. Relation-Aware TGAT" not in html:
    if target_marker in html:
        html = html.replace(target_marker, target_marker + tgat_card_html)
        print("Inserted TGAT card into Glossary Modal!")
    else:
        print("Warning: target_marker not found verbatim.")

with open(deck_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Deck updated with TGAT card!")

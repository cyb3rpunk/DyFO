"""Fix glossary script in bracis_presentation_deck.html."""

from pathlib import Path

deck_path = Path("doc/bracis_presentation_deck.html")

with open(deck_path, "r", encoding="utf-8") as f:
    html = f.read()

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
        if (window.renderMathInElement) {
          renderMathInElement(modal, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false
          });
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

if "function toggleGlossary" not in html:
    # Find the last </script>
    idx = html.rfind("</script>")
    if idx != -1:
        html = html[:idx] + js_to_add + "\n  </script>" + html[idx+9:]
        print("Successfully inserted Glossary JS into main <script> tag!")
    else:
        print("Warning: </script> not found.")

with open(deck_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Deck updated cleanly!")

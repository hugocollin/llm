import time

# Fonction pour afficher le texte progressivement
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)
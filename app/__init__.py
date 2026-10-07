from flask import Flask

# On crée l'objet Flask nommé 'app'
app = Flask(__name__)

# On importe les routes à la FIN pour éviter que routes.py 
# cherche 'app' avant qu'il ne soit créé
from app import routes
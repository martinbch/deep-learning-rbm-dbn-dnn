# TP DNN - AlphaDigit et MNIST

Projet de travaux pratiques consacré aux réseaux de neurones profonds et aux
modèles génératifs appliqués aux jeux de données **Binary AlphaDigits** et
**MNIST**.

## Contenu

- `P4_AlphaDigit.ipynb` : étude des RBM et DBN sur Binary AlphaDigits,
  génération d'images et analyse de l'impact des hyperparamètres.
- `P5_MNIST.ipynb` : classification MNIST avec un DNN, comparaison entre
  initialisation aléatoire et pré-entraînement par RBM/DBN.
- `principal_RBM_alpha.py` : implémentation d'une Restricted Boltzmann
  Machine (RBM).
- `principal_DBN_alpha.py` : implémentation d'un Deep Belief Network (DBN)
  construit à partir de RBM.
- `principal_DNN_MNIST.py` : implémentation du DNN et de son entraînement sur
  MNIST.
- `Rapport_TP_DNN_AlphaDigit_MNIST.pdf` : rapport du projet.

## Prérequis

Python 3.10 ou une version ultérieure est recommandé. Les principales
bibliothèques utilisées sont :

```text
numpy
scipy
matplotlib
tqdm
jupyter
```

Installation rapide :

```bash
pip install numpy scipy matplotlib tqdm jupyter
```

## Données

Les fichiers de données ne sont pas inclus dans ce dépôt. Les chemins vers
Binary AlphaDigits et les fichiers binaires MNIST doivent être configurés
dans les notebooks avant leur exécution.

## Exécution

Depuis la racine du projet :

```bash
jupyter notebook
```

Ouvrir ensuite `P4_AlphaDigit.ipynb` ou `P5_MNIST.ipynb` et exécuter les
cellules dans l'ordre.

## Nom recommandé du dépôt

`TP-DNN-AlphaDigit-MNIST`
# **_Geometric Brownian Motion (GBM)_**

C'est le modèle de base pour simuler les prix d'actifs financiers. Utilisé partout — Black-Scholes, Monte Carlo, stress tests.

## **L'idée intuitive**

**Un prix financier a deux composantes :**
1. Une tendance (drift) — le marché monte en moyenne sur le long terme
2. Un choc aléatoire (diffusion) — chaque jour y'a du bruit imprévisible

## L'équation différentielle stochastique (SDE)

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Décortiquons terme par terme :

- $dS_t$ — variation infinitésimale du prix à l'instant $t$
- $\mu S_t \, dt$ — la partie déterministe (drift). $\mu$ = rendement moyen annualisé. Multiplié par $S_t$ car la tendance est proportionnelle au prix actuel (rendements, pas variations absolues)
- $\sigma S_t \, dW_t$ — la partie stochastique (diffusion). $\sigma$ = volatilité. $dW_t$ = incrément brownien = bruit gaussien pur
- $dW_t$ — c'est le cœur du truc. C'est un incrément d'un mouvement brownien standard :

$$dW_t \sim \mathcal{N}(0,\, dt)$$

```python
# sur un pas discret : W_{t+1} - W_t ~ N(0, 1) × √dt
```

---

## Pourquoi "Géométrique" ?

Parce que le choc est **multiplicatif** (proportionnel à $S_t$), pas additif.

```python
# Brownien standard (additif) — MAUVAIS pour les prix
S_{t+1} = S_t + bruit   →  peut devenir négatif ❌

# Brownien GÉOMÉTRIQUE (multiplicatif) — correct
S_{t+1} = S_t × (1 + bruit)  →  toujours positif ✅
```

Un prix ne peut pas être négatif (enfin, sauf le pétrole en avril 2020 lol, mais c'est un cas extrême).

---

## La solution analytique

On résout la SDE avec le **lemme d'Itô** (le théorème fondamental du calcul stochastique) et on obtient :

$$S_t = S_0 \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$$

Le terme $-\sigma^2/2$ c'est la **correction d'Itô** — c'est ce qui distingue le calcul stochastique du calcul classique. Il vient du fait que $(dW_t)^2 = dt$ (pas $0$ comme en calcul classique).

---

## Version discrète — ce qu'on code

En pratique on discrétise sur des pas journaliers ($\Delta t = 1/252$) :

$$\log\!\left(\frac{S_{t+1}}{S_t}\right) = \underbrace{\left(\mu - \frac{\sigma^2}{2}\right)\Delta t}_{\text{drift}} + \underbrace{\sigma \sqrt{\Delta t} \, \varepsilon_t}_{\text{choc}}, \qquad \varepsilon_t \sim \mathcal{N}(0,1)$$

Ce qui donne en code ce qu'on a fait exactement :

```python
returns = np.random.normal(0.0003, 0.01, n_days)
#                          ^^^^^^  ^^^^
#                          drift   sigma (vol journalière)

prices = 100 * np.exp(np.cumsum(returns))
#                     ^^^^^^^^^^^^^^^^^^
#                     cumsum = somme cumulée des log-rendements
#                     = log(S_t/S_0) à chaque instant t
```

`np.cumsum` c'est la version discrète de l'intégrale $\int dW_t$ — t'additionnes tous les chocs depuis $t=0$.

---

## Les hypothèses (et pourquoi c'est une simplification)

Le GBM suppose que les rendements sont **i.i.d. gaussiens** — indépendants et identiquement distribués. En réalité :

| Hypothèse GBM | Réalité des marchés |
|---|---|
| Rendements gaussiens | Fat tails (queues épaisses) — les crashs arrivent bien plus souvent que prédit |
| Volatilité constante $\sigma$ | Vol clustering — les périodes agitées restent agitées (GARCH) |
| Rendements indépendants | Autocorrélation à court terme, mean-reversion |
| Drift constant $\mu$ | Régimes changeants (bull/bear market) |

C'est pour ça que ton modèle ML a un intérêt — si les marchés étaient vraiment GBM pur, aucun modèle prédictif n'aurait de valeur (hypothèse des marchés efficients).

---

## Le lien avec Black-Scholes

Le GBM c'est exactement l'hypothèse sur laquelle Black & Scholes ont construit leur formule de pricing d'options en 1973 (Nobel 1997). Sous GBM, le prix d'un call européen a une formule fermée :

$$C = S_0 \,\Phi(d_1) - K e^{-rT} \Phi(d_2)$$

où $\Phi$ est la CDF de la gaussienne. Voilà pourquoi le GBM c'est la base de toute la finance quant moderne.

---

# **Résultats — LinearClassifier sur SPY/AAPL/MSFT**

## Données

```
Prix téléchargés : 2515 jours, 3 tickers
[✓] SPY  — 2494 lignes, 13 colonnes
[✓] AAPL — 2494 lignes, 13 colonnes
[✓] MSFT — 2494 lignes, 13 colonnes
```

### Aperçu des features SPY

```
                 close    ret_1d  ...  target_ret_1d  target_clf
Date
2015-02-02  167.218216  0.012384  ...       0.014461           1
2015-02-03  169.636429  0.014461  ...      -0.003808           0
2015-02-04  168.990448 -0.003808  ...       0.010095           1
2015-02-05  170.696457  0.010095  ...      -0.002766           0
2015-02-06  170.224365 -0.002766  ...      -0.004476           0
```

### Colonnes disponibles

```
['close', 'ret_1d', 'ret_lag_1', 'ret_lag_2', 'ret_lag_5',
 'roll_mean_5d', 'roll_mean_10d', 'roll_mean_20d',
 'roll_std_5d', 'roll_std_10d', 'roll_std_20d',
 'target_ret_1d', 'target_clf']
```

```
[✓] Panel complet : (7482, 13)
```

---

## Entraînement

```
[✓] Features utilisées : ['close', 'ret_1d', 'ret_lag_1', 'ret_lag_2', 'ret_lag_5',
                           'roll_mean_5d', 'roll_mean_10d', 'roll_mean_20d',
                           'roll_std_5d', 'roll_std_10d', 'roll_std_20d']
Train on : 1995 days
[✓] Test on  : 499 days

[✓] Modèle : LinearClassifier(
  (linear): Linear(in_features=11, out_features=2, bias=True)
)
[✓] Paramètres : 24
```

### Courbe de loss / accuracy

```
Epoch  10/50 | Loss: 0.7116 | Acc: 0.506
Epoch  20/50 | Loss: 0.6966 | Acc: 0.521
Epoch  30/50 | Loss: 0.6929 | Acc: 0.527
Epoch  40/50 | Loss: 0.6907 | Acc: 0.535
Epoch  50/50 | Loss: 0.6895 | Acc: 0.540
```

---

## Résultats test

```
[✓] Accuracy test : 0.535

              precision    recall  f1-score   support

  Baisse (0)       0.42      0.25      0.32       212
  Hausse (1)       0.57      0.74      0.65       287

    accuracy                           0.54       499
   macro avg       0.50      0.50      0.48       499
weighted avg       0.51      0.54      0.51       499
```

### Importance des features (norme des poids)

```
roll_std_10d     0.1326
roll_mean_5d     0.1283
roll_std_5d      0.1191
roll_mean_10d    0.1110
ret_1d           0.0680
ret_lag_1        0.0648
roll_mean_20d    0.0475
close            0.0420
ret_lag_5        0.0364
ret_lag_2        0.0225
roll_std_20d     0.0205
```

---

## Hypothèse des marchés efficients (EMH)

## **Les prix des actifs financiers reflètent à tout instant toute l'information disponible.**

**Forme semi-forte** — les prix reflètent toute l'information publique (bilans, news, annonces macro)

**Forme forte** — les prix reflètent toute l'information, y compris privée (insider information)

### Le mécanisme qui rend les marchés efficients

C'est une boucle auto-destructrice :

1. Un quant découvre un pattern prédictif dans SPY
2. Il trade dessus → achète quand le signal dit hausse
3. Son trading fait monter le prix **avant** la hausse prévue
4. Le pattern disparaît — il est maintenant dans le prix
5. D'autres quants cherchent → trouvent rien → confirme l'efficience

> Plus un actif est liquide et suivi (SPY, AAPL, MSFT), plus ce mécanisme est rapide et brutal. Un signal sur SPY est arbitragé en **millisecondes** par des HFT.

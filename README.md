# Big Data Project

## Lancement du projet

Pour lancer le projet, vous devez **d'abord créer manuellement** la base de données sous la structure suivante :

```
data/
  ├── euronext/
  └── boursorama/
```

Ensuite, vous devez **modifier** le fichier `docker-compose.yml` pour adapter le chemin dans la section `volumes` en fonction de votre machine :

```yaml
volumes:
  - /votre/chemin/vers/data:/home/bourse/data/
```

Par exemple, le chemin initial était :

```yaml
volumes:
  - /mnt/c/Users/leolo/OneDrive/Bureau/temp_big_data/data:/home/bourse/data/
```

Modifiez-le pour pointer vers votre propre répertoire `data`.

---

## Fonctionnalité supplémentaire : Sélection d'intervalle de dates

Dans le fichier `etl.py`, nous avons ajouté une fonctionnalité permettant à l'utilisateur de **choisir l'intervalle de dates** qu'il souhaite traiter, directement dans `main.py`.

Cela est utile car **calculer toutes les données de 2019 à 2024 est très long** (~20 minutes sur un PC Cisco).  
Ainsi, si l'utilisateur souhaite par exemple uniquement les données de l'année 2019, il peut configurer comme suit :

```python
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)
```

Cela permet de réduire considérablement le temps de traitement.

---

## Instructions de lancement

Après avoir :

- créé la base de données (`data/euronext` et `data/boursorama`),
- modifié correctement le chemin dans `docker-compose.yml`,

vous pouvez **lancer le projet** en exécutant la commande suivante **depuis le dossier `docker`** :

```bash
make all
```

Ensuite, **attendez que l'ETL affiche un code retour 0**, ce qui signifie que le traitement s'est terminé avec succès.  
À ce moment-là, vous pourrez **ouvrir le tableau de bord dans votre navigateur web**.

---

## Remarques sur les performances

- Sur un PC Cisco, le traitement complet (2019-2024) prend environ **20 minutes**.
- Si vous disposez de plus ou moins de RAM, vous pouvez ajuster ce paramètre dans `etl.py` :

```python
commit_threshold = 100000  # À ajuster selon votre RAM : plus c'est élevé, moins il y aura de commits, mais plus cela consommera de mémoire.
```

---

Hésité pas à mettre 20/20 🚀 !

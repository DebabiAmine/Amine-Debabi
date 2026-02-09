# DEBABI Mohamed Amine

**Data Scientist | Data Engineer | AI Engineer**

---

## 1. Qui suis-je

Ingenieur informatique diplome de l'Universite SESAME (2016-2021), specialise en Data Science et Intelligence Artificielle. Avec plus de 5 ans d'experience, j'ai travaille dans des secteurs varies : **banque**, **immobilier**, **conformite reglementaire** et **marketing**.

Mon parcours couvre l'ensemble de la chaine de valeur data :
- **Data Engineering** : conception d'architectures data, pipelines ETL/ELT, modelisation dimensionnelle, data warehousing
- **Data Science & ML** : modeles predictifs, detection d'anomalies, NLP, IA generative, segmentation client
- **Data Quality** : nettoyage, regles metier, conformite, tracabilite

Je transforme les donnees brutes en solutions IA deployees en production, de la collecte jusqu'au monitoring post-deploiement.

**Contact**
- Email : aminemr0194@gmail.com
- Telephone : 0745992224
- LinkedIn : [linkedin.com/in/amine-debabi-53204016a](https://linkedin.com/in/amine-debabi-53204016a)
- GitHub : [github.com/DebabiAmine](https://github.com/DebabiAmine)

**Formation**
- 2018-2021 : Cycle Ingenieur Data Science - Universite SESAME
- 2016-2018 : Cycle Preparatoire Integre - Universite SESAME

**Certifications** : Python (Avance) | Power BI - Data Analysis & Visualization | Insights from Data with BigQuery | BigQuery for Data Warehousing

**Langues** : Francais (maternelle) | Arabe (maternelle) | Anglais (technique)

---

## 2. Projets Data Science & Machine Learning

Cette section presente mes projets sous l'angle Data Science : le processus complet depuis l'exploration des donnees jusqu'au deploiement des modeles.

---

### 2.1 Extraction Intelligente de Factures - XFlowData

*CDI | Fevrier 2026 - Present*

**Objectif** : Developper une plateforme pour recuperer les donnees des factures multi-formats (PDF, PNG, JPEG, TIFF) et les stocker dans une base de donnees structuree.

**Etape 1 - Ingestion & OCR**
- Conception d'un pipeline OCR avance combinant Tesseract et modeles de Deep Learning pour la reconnaissance de texte sur documents scannes et photos
- Traitement multi-formats : PDF natifs, images scannees (PNG, JPEG, TIFF), documents photographies

**Etape 2 - Extraction structuree via LLM**
- Utilisation du LLM Qwen2.5:3b pour l'extraction intelligente des champs cles : numero de facture, date, montant HT/TTC, TVA, fournisseur, lignes de detail
- Prompting structure pour guider le LLM a extraire les donnees dans un format JSON normalise
- Classification automatique des types de documents (facture, avoir, devis) via le LLM
- Combinaison OCR + LLM : le texte brut extrait par OCR est envoye au LLM pour structuration et interpretation semantique

**Etape 3 - Validation & Regles metier**
- Regles de validation : coherence montants, verification TVA, detection des doublons
- Controle de qualite des extractions avec taux de confiance par champ
- Post-traitement LLM : verification croisee entre les champs extraits (total = somme des lignes, TVA coherente)

**Etape 4 - Stockage & API**
- Conception du modele de donnees relationnel pour le stockage structure des factures extraites dans PostgreSQL
- Developpement d'API REST (FastAPI) pour l'integration avec les systemes comptables existants

**Etape 5 - Monitoring**
- Dashboards de monitoring : taux de reconnaissance, taux d'erreur, volume traite, temps de traitement

`Python` `OCR (Tesseract)` `Qwen2.5:3b` `LLM` `FastAPI` `PostgreSQL` `Docker`

---

### 2.2 Prediction des prix immobiliers - Century21 Tunisie

*Freelance | Juillet 2025 - Janvier 2026*

**Objectif** : Developper un systeme de prediction des prix immobiliers en Tunisie a partir de donnees multi-sources.

**Etape 1 - Exploration & Analyse (EDA)**
- Analyse de la distribution des variables (prix, surface, localisation, nombre de pieces)
- Matrices de correlation pour identifier les relations entre variables
- Detection de patterns saisonniers et tendances geographiques
- Identification des outliers via visualisations (boxplots, scatter plots)

**Etape 2 - Nettoyage & Preparation**
- Suppression des doublons (`drop_duplicates`) et gestion des valeurs manquantes
- Detection et traitement des outliers avec IQR et Isolation Forest
- Imputation avancee des valeurs manquantes : KNN Imputer, MICE (Multiple Imputation by Chained Equations)

**Etape 3 - Feature Engineering**
- Creation de variables derivees comme : ratio surface/piece, proximite transports, indice de quartier , NOI , Zipcode ,MSA 
- Encodage des variables categorielles : Target Encoding, One-Hot Encoding 
- Transformation de features temporelles : anciennete du bien, saisonnalite du marche
- Normalisation et scaling des variables numeriques

**Etape 4 - Modelisation & Entrainement**
- Developpement et comparaison de modeles : XGBoost, LightGBM, Random Forest
- Validation croisee stratifiee (K-Fold) pour garantir la robustesse et eviter l'overfitting
- Optimisation des hyperparametres avec GridSearchCV et RandomizedSearchCV
- Reduction de l'erreur de prediction de 15% apres tuning

**Etape 5 - Evaluation & Metriques**
- Metriques de regression : MAE, MAPE, RMSE, R2
- Analyse comparative des performances entre modeles
- Intervalles de confiance sur les predictions

**Etape 6 - Interpretabilite**
- SHAP values pour identifier les drivers de prix (localisation, surface, etage, etc.)
- Feature importance globale et locale pour chaque prediction
- Rapports d'explicabilite pour les equipes metier

**Etape 7 - Deploiement & Monitoring**
- Mise en production avec MLflow : tracking des experimentations, versioning des modeles, registry centralise
- Deploiement sur Snowflake Cortex AI avec monitoring en temps reel
- Detection automatique du data drift et model drift
- API FastAPI pour exposition des predictions avec systeme de cache et requetes batch
- Pipelines d'entrainement automatises avec re-training periodique
- Dashboards de monitoring ML avec alertes automatiques sur degradation des performances

`Python` `Scikit-learn` `XGBoost` `LightGBM` `MLflow` `SHAP` `Snowflake Cortex AI` `FastAPI` `Pandas` `NumPy`

---

### 2.3 Prediction des prix immobiliers - Valbridge Property Advisors

*Freelance | Janvier 2024 - Juin 2025*

**Objectif** : Construire un systeme de prediction immobiliere pour le marche americain avec quantification de l'incertitude.

**Etape 1 - Exploration & Analyse (EDA)**
- Analyse exploratoire approfondie : distributions, correlations, detection de patterns
- Visualisations avancees pour comprendre les dynamiques du marche

**Etape 2 - Feature Engineering**
- Variables geographiques : distance aux centres d'interet, scores de quartier
- Agregations temporelles : tendances de prix par periode et zone
- Ratios metier pertinents pour le secteur immobilier

**Etape 3 - Modelisation**
- Comparaison de modeles Gradient Boosting : LightGBM, XGBoost, CatBoost
- Ensemble methods : Stacking et Blending pour combiner les forces de chaque modele
- Calibration des modeles et analyse des intervalles de confiance

**Etape 4 - Industrialisation**
- Pipelines automatises d'entrainement et de scoring sur GCP BigQuery ML
- API REST (FastAPI) avec documentation Swagger, tests unitaires, monitoring des temps de reponse
- Suivi des KPIs et accompagnement client dans l'exploitation des resultats

`Python` `LightGBM` `XGBoost` `CatBoost` `Scikit-learn` `GCP BigQuery ML` `FastAPI` `MLflow` `Matplotlib` `Seaborn`

---

### 2.4 Detection et prevention des fraudes AML/FT - Tanit Lab

*CDI | Novembre 2023 - Mars 2025*

**Objectif** : Developper un systeme de detection des transactions frauduleuses pour la conformite anti-blanchiment.

**Etape 1 - Comprehension metier**
- Recueil des besoins avec les equipes Risque & Conformite (normes SNI, IFRS9)
- Definition des regles metier : seuils par region, par secteur d'activite, par montant
- Identification des scenarios de fraude connus et des patterns suspects

**Etape 2 - Feature Engineering transactionnel**
- Construction de profils transactionnels : montants moyens, frequence des operations, ecarts aux habitudes
- Agregations temporelles : comportement sur 7j, 30j, 90j
- Ratios comportementaux : ecart par rapport au profil habituel du client
- Variables metier : secteur d'activite, region geographique, type de transaction

**Etape 3 - Gestion du desequilibre des classes**
- Les fraudes sont des evenements rares (<1% des transactions)
- Application de SMOTE (Synthetic Minority Over-sampling Technique) pour generer des exemples synthetiques
- Undersampling de la classe majoritaire
- Ajustement des seuils de classification pour optimiser le compromis precision/rappel

**Etape 4 - Modelisation**
- Modeles de detection d'anomalies : Isolation Forest, LOF (Local Outlier Factor), Autoencoders
- Systeme de scoring multi-criteres avec ponderation adaptative selon les regles reglementaires
- Validation sur donnees historiques et simulation de scenarios de fraude

**Etape 5 - Metriques specifiques**
- Precision-Recall curve (plus pertinente que ROC pour classes desequilibrees)
- F2-Score : ponderation en faveur du rappel (ne pas rater de fraude)
- Taux de detection, taux de faux positifs, cout metier des erreurs

**Etape 6 - Explicabilite & Conformite**
- LIME (Local Interpretable Model-agnostic Explanations) pour expliquer chaque alerte individuellement
- SHAP values pour comprendre les facteurs globaux de risque
- Justification des decisions pour les equipes Conformite et les regulateurs
- Tracabilite complete des scores pour conformite reglementaire

**Etape 7 - Monitoring post-deploiement**
- Suivi continu des taux de detection et faux positifs
- Feedback loop : integration des retours des analystes pour amelioration continue
- Rapports KPI automatises avec Power BI

`PySpark` `HDFS` `PostgreSQL` `Isolation Forest` `SMOTE` `SHAP` `LIME` `Power BI` `GitLab CI/CD`

---

### 2.5 Chatbot IA - Secteur bancaire - Tanit Lab

*CDI | Novembre 2023 - Mars 2025*

**Objectif** : Creer un chatbot intelligent pour le service client bancaire base sur l'architecture RAG et le fine-tuning de LLMs.

**Etape 1 - Analyse des besoins**
- Identification des cas d'usage conversationnels avec le Service Client et la Conformite
- Priorisation des scenarios : FAQ produits, procedures bancaires, reglementations

**Etape 2 - Preparation du corpus documentaire**
- Collecte des documents bancaires : FAQ, documentation produits, textes reglementaires
- Nettoyage des documents : suppression du bruit (headers, footers, pagination)
- Standardisation des formats : conversion PDF, Word vers texte structure
- Segmentation en sections thematiques pour un contexte coherent

**Etape 3 - Architecture RAG (Retrieval-Augmented Generation)**
- Chunking intelligent des documents : decoupe en segments semantiquement coherents
- Vectorisation avec des modeles d'embedding (sentence-transformers)
- Indexation des vecteurs dans ChromaDB pour recherche par similarite
- Pipeline LangChain : requete utilisateur -> retrieval des chunks pertinents -> generation de reponse contextuelle

**Etape 4 - Fine-tuning du LLM**
- Fine-tuning de LLaMA 3 (8B parametres) via LoRA (Low-Rank Adaptation)
- Corpus d'entrainement : paires question/reponse issues du domaine bancaire
- Tests de robustesse NLP : coherence, pertinence, gestion des cas limites

**Etape 5 - Metriques LLM & RAG**
- Relevance score : pertinence des documents recuperes par le retriever
- Answer quality : qualite et precision des reponses generees
- Hallucination rate : taux de reponses inventees vs basees sur les sources
- Latency : temps de reponse du pipeline complet
- Tests utilisateurs avec monitoring des metriques cles

`LangChain` `LLaMA 3` `LoRA` `ChromaDB` `Hugging Face` `RAG` `TensorFlow` `sentence-transformers`

---

### 2.6 Assistant Intelligent de Recherche Documentaire (RAG) - Tanit Lab

*CDI | Novembre 2023 - Mars 2025*

**Objectif** : Systeme de recherche documentaire intelligent capable d'interroger une base de connaissances multi-formats.

**Etape 1 - Ingestion multi-formats**
- Pipeline de traitement : PDF, Word, Excel avec extraction et structuration automatique
- Nettoyage du contenu : suppression des artefacts, normalisation du texte

**Etape 2 - Strategies de chunking**
- Semantic chunking : decoupe basee sur le sens des paragraphes
- Sliding window : chevauchement entre chunks pour ne pas perdre le contexte
- Hierarchisation des sections : metadata (titre, chapitre) preservees pour chaque chunk

**Etape 3 - Embedding & Indexation**
- Experimentation comparative : sentence-transformers vs OpenAI embeddings
- Analyse des performances de chaque modele d'embedding sur le corpus
- Indexation vectorielle dans ChromaDB avec optimisation des parametres

**Etape 4 - Retrieval & Re-ranking**
- Recherche par similarite cosinus dans ChromaDB
- Re-ranking avec modeles cross-encoder pour affiner la pertinence des documents
- Scoring et ranking des sources pour tracabilite et fiabilite

**Etape 5 - Generation multi-LLM**
- Integration de GPT-4 et LLaMA 3 via LangChain
- Routing intelligent : selection du LLM selon le type de requete (factuel, analytique, creatif)
- Prompts optimises pour chaque cas d'usage

**Etape 6 - Evaluation**
- Relevance score, answer quality, hallucination rate
- Optimisation de la latence du pipeline end-to-end

`LangChain` `GPT-4` `LLaMA 3` `ChromaDB` `RAG` `Hugging Face` `sentence-transformers` `cross-encoder`

---

### 2.7 Smarketys - Optimisation marketing - Tanit Lab

*CDI | Novembre 2023 - Mars 2025*

**Objectif** : Optimiser l'allocation budgetaire marketing et segmenter les clients pour personnaliser les offres.

**Etape 1 - Marketing Mix Modeling (MMM)**
- Developpement de modeles MMM avec Robyn pour attribution multi-touch
- Modelisation des effets adstock et carryover : impact differe des campagnes dans le temps
- Courbes de saturation : identification du point de rendement decroissant par canal
- Calcul du ROI marginal par canal pour recommandations d'allocation budgetaire

**Etape 2 - Segmentation client**
- Analyse RFM (Recency, Frequency, Monetary) pour profiler les clients
- Algorithmes de clustering : K-Means, DBSCAN, Hierarchical Clustering
- Segmentation comportementale : patterns d'achat, preferences, canaux preferes
- Modelisation de Customer Lifetime Value (CLV)

**Etape 3 - Interpretabilite & Recommandations**
- Feature importance pour identifier les leviers d'optimisation marketing
- Recommandations de personnalisation des offres par segment
- Dashboards Power BI pour suivi et analyse des performances

**Etape 4 - OCR**
- Module OCR (PyTesseract) pour extraction depuis tickets de caisse et formulaires papier
- Integration des donnees offline dans le pipeline d'analyse

`Python` `MMM` `Robyn` `K-Means` `DBSCAN` `CLV` `Power BI` `OCR (Tesseract)` `SQL`

---

## 3. Projets Data Engineering

Cette section presente mes projets sous l'angle Data Engineering : architecture, pipelines, modelisation et orchestration des flux de donnees.

---

### 3.1 Architecture Data sur Snowflake - Century21 Tunisie

*Freelance | Juillet 2025 - Janvier 2026*

**Objectif** : Concevoir une plateforme data centralisee pour l'immobilier en Tunisie.

**Etape 1 - Architecture Medallion**
- Design du Data Warehouse avec approche medallion :
  - **Bronze** : donnees brutes ingestees telles quelles depuis les sources
  - **Silver** : donnees nettoyees, transformees, standardisees
  - **Gold** : donnees enrichies, agregees, pretes pour l'analyse et les modeles ML
- Separation des environnements : dev / staging / prod

**Etape 2 - Modelisation dimensionnelle**
- Schemas en etoile : tables de faits (transactions immobilieres) et dimensions (geographie, temps, proprietes, agents)
- Optimisation des jointures et des requetes analytiques

**Etape 3 - Pipelines ELT**
- Ingestion depuis sources multiples : APIs REST, fichiers CSV/JSON, web scraping
- Developpement de scripts Python + Selenium pour scraping automatise multi-sources
- Orchestration des flux avec versioning

**Etape 4 - Change Data Capture (CDC)**
- Snowflake Streams pour capturer les modifications en temps reel
- Tracking des insertions, mises a jour et suppressions

**Etape 5 - Data Quality & Monitoring**
- Contraintes d'integrite, validation de formats, detection de duplicatas
- Alertes automatiques sur anomalies de donnees
- Dashboards de monitoring (Snowflake, Power BI)

**Etape 6 - Documentation**
- Diagrammes ERD, dictionnaire de donnees
- Runbooks pour operations, guides d'utilisation pour analystes

`Snowflake` `Azure` `Python` `Selenium` `FastAPI` `SQL` `Git` `Power BI`

---

### 3.2 Data Warehouse sur GCP BigQuery - Valbridge Property Advisors

*Freelance | Janvier 2024 - Juin 2025*

**Objectif** : Construire un Data Warehouse performant pour les donnees immobilieres americaines.

**Etape 1 - Modelisation**
- Schema etoile sur GCP BigQuery
- Partitionnement par date pour optimisation des couts
- Clustering sur colonnes cles pour accelerer les requetes

**Etape 2 - Pipelines ETL**
- Extraction depuis sources heterogenes : APIs, fichiers Excel/CSV, bases SQL
- Transformations Python : nettoyage, standardisation, enrichissement
- Chargement batch et streaming vers BigQuery

**Etape 3 - Data Quality**
- Tests automatises avec great_expectations
- Validation de schemas, detection de derives
- Alertes sur anomalies de donnees

**Etape 4 - Industrialisation**
- Pipelines automatises d'entrainement et de scoring sur BigQuery ML
- API FastAPI pour acces temps reel aux resultats

`GCP BigQuery` `Python` `great_expectations` `FastAPI` `SQL` `Git`

---

### 3.3 Stack ELK & ETL Talend - BIAT

*CDI | Juillet 2021 - Octobre 2023*

**Objectif** : Centraliser les donnees clients de la banque pour le ciblage des offres digitales MyBIAT.

**Etape 1 - Modelisation relationnelle**
- Schemas normalises dans Oracle : clients, comptes, transactions, produits
- Contraintes d'integrite referentielle entre les tables
- Dictionnaire de donnees et diagrammes de flux

**Etape 2 - Pipelines ETL avec Talend**
- **Extraction** : connexion a Oracle (core banking) avec les composants Talend (tOracleInput)
- **Transformation** : tMap pour les jointures complexes, les filtres, les lookups et le mapping entre schemas source et cible
- **Chargement** : ecriture vers Elasticsearch pour indexation et recherche rapide
- Jobs schedules avec gestion des erreurs, logging detaille, notifications d'echecs
- Rejet des donnees invalides avec tracabilite

**Etape 3 - Stack ELK**
- **Logstash** : ingestion temps reel depuis les sources
- **Elasticsearch** : stockage et recherche full-text, mapping des documents, sharding et replication pour haute disponibilite
- **Kibana** : dashboards interactifs pour l'exploitation des donnees consolidees

**Etape 4 - Automatisation & Monitoring**
- Jobs Talend automatises et planifies
- Data quality checks : validation de formats, detection de doublons, coherence entre systemes sources
- Rapports d'anomalies et procedures d'exploitation pour les equipes IT

`Talend` `tMap` `Oracle XE` `Elasticsearch` `Logstash` `Kibana` `SQL` `Python`

---

## 4. Data Quality & Regles Metier

Cette section detaille mon approche de la qualite des donnees : comment je nettoie, structure et adapte les donnees brutes aux contraintes metier avant de les injecter dans les modeles IA.

---

### 4.1 Fraude AML/FT - Nettoyage & Regles Metier

**Problematique** : Les donnees transactionnelles bancaires sont bruyantes, heterogenes et doivent respecter des contraintes reglementaires strictes avant d'alimenter un modele de detection de fraude.

**Nettoyage des donnees**
- Extraction automatisee depuis factures et documents financiers via OCR (Tesseract)
- Suppression des doublons transactionnels (meme montant, meme compte, meme timestamp)
- Gestion des valeurs manquantes : imputation ou exclusion selon le champ
- Standardisation des formats : dates, montants, devises, codes pays

**Regles metier appliquees**
- **Par region** : seuils de montant differents selon la zone geographique, detection des flux cross-border suspects
- **Par secteur d'activite** : profils de transaction attendus par secteur (commerce, immobilier, services), alertes sur ecarts significatifs
- **Par montant** : regles de seuil (declarations obligatoires au-dela de certains montants), detection du fractionnement (structuring)
- **Par profil client** : comparaison du comportement actuel vs historique, detection des changements brusques de pattern

**Adaptation pour le modele IA**
- Les regles metier (BR) seules ne suffisent pas : elles detectent les cas connus mais pas les nouveaux schemas de fraude
- Le modele IA (Isolation Forest, Autoencoders) apprend les patterns normaux et detecte les deviations inconnues
- Combinaison BR + IA : les regles metier servent de features supplementaires et de filtres de pre-selection, le modele IA generalise au-dela des regles fixes
- Ponderation adaptative : les poids des criteres metier s'ajustent selon le contexte reglementaire

---

### 4.2 Chatbot bancaire - Nettoyage & Preparation des documents

**Problematique** : Les documents bancaires (FAQ, reglementations, fiches produits) sont dans des formats heterogenes et contiennent du bruit qui degrade la qualite du RAG.

**Nettoyage des documents**
- Conversion multi-formats : PDF, Word, Excel vers texte brut structure
- Suppression du bruit : headers/footers repetitifs, numeros de page, watermarks, metadata inutiles
- Normalisation du texte : correction des encodages, uniformisation de la ponctuation, suppression des espaces multiples
- Detection et suppression des sections non pertinentes (mentions legales generiques, pages blanches)

**Structuration pour le RAG**
- Identification des sections thematiques : produits, procedures, reglementations
- Preservation des metadata : titre du document, date, categorie, source
- Chunking semantique : decoupe en segments qui preservent le sens complet d'une idee
- Validation de la coherence : chaque chunk doit etre auto-suffisant et comprehensible hors contexte

**Regles metier documentaires**
- Hierarchie d'autorite : les textes reglementaires priment sur les FAQ internes
- Versioning : seule la derniere version d'un document est indexee
- Confidentialite : exclusion des documents a diffusion restreinte du corpus public

---

### 4.3 Donnees immobilieres - Nettoyage & Standardisation

**Problematique** : Les donnees immobilieres proviennent de sources multiples (scraping, APIs, fichiers) avec des formats et des qualites tres variables.

**Nettoyage**
- Deduplication : detection des annonces en double (meme bien sur plusieurs plateformes)
- Outliers : biens a prix aberrants (erreurs de saisie), surfaces incoherentes
- Valeurs manquantes : imputation KNN/MICE pour les features numeriques, mode pour les categorielles
- Standardisation : unites (m2, pieces), formats d'adresse, nomenclature des types de bien

**Regles metier immobilieres**
- **Par region** : fourchettes de prix attendues par zone, detection des anomalies geographiques
- **Par type de bien** : ratios coherents surface/prix, nombre de pieces/surface
- **Par temporalite** : exclusion des annonces obsoletes, detection des prix non actualises

**Data quality checks automatises**
- great_expectations pour validation continue des schemas et des valeurs
- Alertes automatiques sur anomalies de donnees
- Rapports de qualite avec taux de completude, taux de doublons, distribution des valeurs

---

## 5. Stack Technique

### Langages & Frameworks
`Python` `PySpark` `R` `SQL` `FastAPI`

### Machine Learning & Deep Learning
`Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `TensorFlow` `PyTorch` `MLflow`

### IA Generative & NLP
`LangChain` `LLaMA 3` `GPT-4` `Hugging Face` `RAG` `ChromaDB` `LoRA` `sentence-transformers`

### Interpretabilite & Metriques
`SHAP` `LIME` `MAPE` `MAE` `RMSE` `F2-Score` `Precision-Recall`

### Big Data & Bases de donnees
`Snowflake` `GCP BigQuery` `PostgreSQL` `Oracle` `MongoDB` `Elasticsearch` `Spark` `Hadoop` `HDFS` `Kafka` `GraphQL`

### Cloud & Deploiement
`Azure` `GCP` `Snowflake Cortex AI` `BigQuery ML`

### ETL & Data Quality
`Talend (tMap, tOracleInput)` `Logstash` `great_expectations` `Selenium` `OCR (Tesseract)`

### Visualisation & BI
`Power BI` `Kibana` `Power Automate` `Matplotlib` `Seaborn`

### DevOps & Gestion de projet
`Git` `GitLab CI/CD` `Jira` `Slack` `Scrum`

### Marketing & Segmentation
`MMM (Marketing Mix Modeling)` `Robyn` `CLV` `RFM` `K-Means` `DBSCAN`

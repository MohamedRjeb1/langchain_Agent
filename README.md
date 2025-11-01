# Local YouTube → RAG (fully local) 🛠️

Pipeline local complet pour interroger le contenu de vidéos YouTube via RAG, sans services cloud:
- yt-dlp + ffmpeg pour l’audio
- Whisper pour la transcription (local)
- Chunking sémantique dynamique avec seuil adaptatif
- Embeddings locaux via Ollama (nomic-embed-text)
- Index FAISS optimisé (cosine via L2)
- LLM local (Mistral via Ollama)
- Corrective RAG (auto-grade + query rewrites)
- Mémoire pour améliorer la récupération (sans injecter la mémoire dans le prompt)
- UI Streamlit “chat” avec panneau de logs enrichi à droite


## Aperçu rapide

- Interface chat Streamlit: à gauche le chat, à droite un large panneau de logs détaillant l’ingestion, le chunking, l’indexation et la récupération.
- Ingestion YouTube → audio mp3 → transcription Whisper → chunking sémantique → embeddings Ollama → index FAISS + clustering (KMeans + silhouette).
- Récupération: FAISS + léger rerank + diagnostics (faiss_similarity, rank, cluster_id) + Corrective RAG (grading + réécritures) + résumé de preuves si trop long.
- Mémoire: les requêtes et les Q/R sont vectorisées et utilisées pour booster la récupération (centroïde + sous-requêtes). La mémoire n’est pas injectée textuellement dans le prompt final.


## Architecture et services

Le code se trouve principalement sous `app/services/`.

- YouTubeService (`youtube_service.py`)
	- Télécharge l’audio via yt-dlp, applique une extraction FFmpeg en mp3, prend en charge un `progress_hook` pour les logs de progression.
	- Sorties: chemin du fichier audio, titre/metadata, statut.

- TranscriptionService (`transcription_service.py`)
	- Transcrit l’audio en texte via Whisper.
	- Utilise `ModelLoaderService` pour charger et mettre en cache le modèle (typiquement `small` par défaut) et journalise les étapes.
	- Sauvegarde le transcript dans `data/transcripts/{task_id}_transcript.txt`.

- SemanticChunkingService (`semantic_chunking_service.py`)
	- Chunking sémantique dynamique avec:
		- Seuil de base: 0.65
		- Seuil adaptatif basé sur la variance locale des similarités inter-segments
		- Taille min: 100 caractères; taille max: 2000 caractères
	- Similarité mesurée via embeddings (LocalEmbeddingService) + cosinus.
	- Journalisation fine: progression, similarité entre segments consécutifs (bornée pour éviter le spam sur très longs textes), finalisation de chaque chunk avec stats (segments, caractères, seuil).

- LocalEmbeddingService (`embedding_service.py`)
	- Embeddings locaux via Ollama (`nomic-embed-text`, dimension 768), avec un petit cache.
	- `embed_query` et `embed_documents` exposés; messages d’erreur non bloquants.

- FAISSIndex (`faiss_index.py`)
	- Index FAISS `IndexFlatIP` avec normalisation L2 (cosine ≈ inner product après L2).
	- Scores normalisés dans [0,1]. Persistence via `faiss.write_index` + JSON pour la map doc.
	- Verrous de fichier pour éviter les corruptions concurrentes; reconstruction de vecteurs pour un léger reranking.

- AdvancedIndexingService (`indexing_service.py`)
	- Construction d’un index FAISS à partir des embeddings et des objets documents.
	- Clustering sémantique via KMeans (n_init=10) + silhouette; clusters et métadonnées persistés.
	- `search_similar(...)`:
		- Cherche dans FAISS, conserve `faiss_similarity`, calcule un sim cohérent via vecteurs reconstruits.
		- Retourne: `content`, `metadata`, `similarity_score`, `faiss_similarity`, `cluster_id`, `rank`.
	- `get_index_statistics(...)`, `load_index(...)`, `search_by_cluster(...)` (recherche textuelle simple par cluster).

- LLMService (`llm_service.py`)
	- Génération via `langchain_ollama.OllamaLLM` (Mistral) avec repli HTTP si besoin.
	- Paramètres: `max_tokens`, `temperature`.

- CombinedRAGService (`rag_service.py`)
	- Ingestion de bout en bout: download → transcribe → chunk → embed → index.
	- Réponse à une requête: construit un retriever FAISS, exécute la récupération + Corrective RAG + mémoire.
	- Corrective RAG: grading LLM des “strips” récupérés (robuste aux sorties JSON encapsulées), réécritures si le meilleur grade < seuil, puis re‑grade.
	- Mémoire: stockage de la requête et de la Q/R, réutilisation pour booster la récupération (fusion centroïde et sous‑requêtes) uniquement; pas d’injection textuelle en prompt.
	- Résumé optionnel des preuves si le texte dépasse un budget (compression via LLM avant réponse).

- RouterService (`router_service.py`)
	- Classifieur (LLM ou heuristiques) “RAG” vs “NORMAL”. L’UI actuelle force “RAG” pour chaque réponse; le router reste disponible si besoin.


## Stratégies de chunking

Le chunking est sémantique et dynamique:
- On segmente en phrases/paragraphes, puis on calcule la similarité cosinus entre segments consécutifs via les embeddings locaux.
- Seuil adaptatif: on abaisse/hausse légèrement le seuil de base (0.65) selon la variance locale des similarités (fenêtre récente), borné à [0.3, 0.9].
- Fin d’un chunk si: similarité < seuil adaptatif ou si la taille max (2000 chars) serait dépassée; on respecte aussi une taille minimale (100 chars) pour valider un chunk.
- Chaque chunk porte des métadonnées: `task_id`, `chunk_id`, `chunk_index`, `segment_count`, `character_count`, `similarity_threshold_used`, etc.


## Stratégies de retrieval

1) FAISS (cosine normalisée): on récupère k candidats avec similarité ∈ [0,1] (FAISS).
2) Rerank léger: on reconstruit les vecteurs des top-k et on recalcule un cosinus exact vs le vecteur de requête normalisé (toujours borné dans [0,1]).
3) Diagnostics attachés pour la transparence:
	 - `similarity_score` (post-rerank), `faiss_similarity` (score FAISS initial), `rank` (rang final), `cluster_id` (si clustering existant).
4) Mémoire pour le boost de précision (jamais injectée au prompt):
	 - Fusion centroïde (requête + centroid des souvenirs proches) → recherche directe par vecteur.
	 - Sous-requêtes dérivées des souvenirs Q/A les plus proches (multi-query retrieval).
5) Corrective RAG (CRAG):
	 - Grading LLM des strips (sortie JSON robuste aux ```json … ``` et aux textes parasites).
	 - Si le meilleur grade < seuil (ex: 0.6), on génère N réécritures, on relance des recherches, on fusionne + re‑grade.
	 - On retient ensuite les top-k par (grade, similarity).


## UI Streamlit (chat + logs)

- Chat à gauche (toujours RAG), logs à droite (colonne plus large pour mieux lire les diagnostics).
- Ingestion: saisissez l’URL YouTube et un “Video title” (servira de `task_id`). Suivez la progression dans les logs (download, transcription, chunking, embeddings, index, clustering).
- Chat: posez vos questions sur la vidéo. Les logs montrent les top résultats avec rang, similarités (FAISS vs rerank), cluster et id de chunk.
- Par défaut, la “provenance” n’est pas affichée dans le chat (elle reste disponible dans les logs).


## Structure du projet

```
app/
	core/config.py                # Config pydantic (chemins, modèles par défaut, etc.)
	services/
		youtube_service.py          # yt-dlp + ffmpeg → mp3, avec hooks de progression
		transcription_service.py    # Whisper, persistance transcript
		semantic_chunking_service.py# Chunking sémantique dynamique + logs
		embedding_service.py        # Ollama embeddings (nomic-embed-text)
		faiss_index.py              # Wrapper FAISS (L2, save/load, reconstruct)
		indexing_service.py         # Build/Load FAISS + clustering + search_similar
		llm_service.py              # Ollama LLM (Mistral) avec fallback HTTP
		rag_service.py              # Orchestration ingestion + RAG + mémoire + CRAG
		router_service.py           # Router (non utilisé par l’UI actuelle)

streamlit_app.py                # UI chat + logs (colonne droite élargie)
data/
	audio/                        # mp3 téléchargés
	transcripts/                  # {task_id}_transcript.txt
	VectorStore/                  # {task_id}.faiss, {task_id}_docs.json, metadata
```


## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé et en cours d’exécution
	- Modèles requis: `mistral` (ou un modèle Mistral local équivalent) et `nomic-embed-text`
- ffmpeg installé (pour yt-dlp → mp3)


## Installation (Windows PowerShell)

```powershell
# 1) Cloner ce dépôt (adapté si vous n’êtes pas déjà dans le dossier)
# git clone <repository-url>
# cd la

# 2) Créer et activer l’environnement
python -m venv venv
venv\Scripts\Activate

# 3) Installer les dépendances Python
pip install -r requirements.txt

# 4) S’assurer qu’Ollama tourne et que les modèles sont présents
# (dans un autre terminal)
# ollama serve
# ollama pull mistral
# ollama pull nomic-embed-text

# 5) ffmpeg doit être accessible dans le PATH
# (optionnel) winget install Gyan.FFmpeg

# 6) Lancer l’UI
streamlit run streamlit_app.py
```

Variables de configuration optionnelles via `.env` (chargé par pydantic):

```
DEFAULT_LLM_MODEL=mistral
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
WHISPER_MODEL=small
WHISPER_LANGUAGE=en
VECTORSTORE_DIR=data/VectorStore
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```


## Guide de test simple

1) Lancez l’UI Streamlit: la page s’ouvre dans le navigateur.
2) Saisissez une URL YouTube et un “Video title” (ex: `video_1`), cliquez “Start ingestion”.
	 - Surveillez la colonne “Logs”: progression du download, transcription Whisper, chunking sémantique (sim, seuils, finalisation), embeddings, index FAISS, clustering.
3) Une fois l’ingestion terminée, posez des questions dans le chat.
	 - Les logs montrent les diagnostics de récupération: rang, similitudes FAISS vs rerank, cluster, chunk_id.
	 - Si le grading JSON échoue, un log explique que des notes 0.00 sont utilisées par défaut; sinon vous verrez des grades non nuls.
4) Relancez des questions: la mémoire (queries/QA) va progressivement améliorer la récupération via centroïde et sous‑requêtes (sans polluer le prompt).


## Dépannage

- “OllamaLLM not available” ou erreurs de génération: vérifiez que `langchain_ollama` est installé et qu’Ollama tourne (modèles pull).
- “faiss non installé”: installez `faiss-cpu` correspondant à votre plateforme.
- yt-dlp/ffmpeg: vérifiez que `ffmpeg` est installé et accessible dans le PATH.
- Whisper modèle: le premier run peut télécharger le modèle; si nécessaire utilisez la fonction de préchargement ou déclenchez une transcription pour le mettre en cache.
- Grades toujours à 0.00: c’était souvent dû à des sorties JSON encapsulées par le LLM; le parsing est maintenant robuste aux ```json ...``` et au texte additionnel. Vérifiez les logs côté “[llm] Grading …” pour voir l’état.


## Licence

Usage et redistribution sous la licence du projet.


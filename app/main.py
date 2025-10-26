
from app.services.semantic_chunking_service import SemanticChunkingService



if __name__ == "__main__":
    
    sample_text = (
    "Bonjour. Ceci est un test. "
    "Le modèle doit segmenter ce texte en phrases. "
    "Ensuite, il doit agréger des phrases similaires pour créer des chunks cohérents. "
    "Ce paragraphe parle de développement web et machine learning. "
    "La programmation Python est au coeur du ML. "
    "Enfin, on vérifie que le chunking retourne des chunks et des métadonnées."
)

    svc = SemanticChunkingService()
    chunks = svc.create_semantic_chunks(sample_text, task_id="test_task")
    print("chunks:", len(chunks))
    for c in chunks:
        print(c["metadata"]["chunk_id"], c["metadata"]["word_count"])

    

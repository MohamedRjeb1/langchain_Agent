from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class SummarisingService:
    def __init__(self, model_name: str = "t5-large", device: str = None):
        """
        Initialise le tokenizer et le modèle.
        device: 'cuda' pour GPU, 'cpu' par défaut.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Détection automatique du device si non fourni
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)

    def summarize_chunks(self, chunks: list[str], max_length: int = 150, min_length: int = 50, num_beams: int = 4) -> str:
        """
        Résume une liste de textes (chunks) et retourne un résumé combiné.
        chunks: liste de textes à résumer
        max_length: longueur maximale de chaque résumé
        min_length: longueur minimale de chaque résumé
        num_beams: nombre de beams pour la génération (plus élevé = résumé plus précis)
        """
        summaries = []

        for chunk in chunks:
            # Tokenization
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Génération du résumé
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True
            )

            # Décodage
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Combinaison de tous les résumés
        final_summary = " ".join(summaries)
        return final_summary

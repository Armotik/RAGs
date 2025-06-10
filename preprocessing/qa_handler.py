import torch
from transformers import pipeline


# La fonction globale generate_qa_from_text est maintenant remplacée par
# une classe pour mieux gérer l'assignation du modèle à un GPU spécifique.

class QAGenerator:
    """
    Une classe pour générer des paires de questions-réponses à l'aide d'un modèle 
    Transformers sur un appareil spécifique (GPU/CPU).
    """

    def __init__(self, device):
        """
        Initialise le générateur de QA sur un appareil spécifique.

        Args:
            device (str): L'appareil sur lequel charger le modèle, par exemple 'cuda:0' ou 'cpu'.
        """
        self.device = device
        # On charge le pipeline sur le GPU assigné.
        # Le paramètre device garantit que le modèle et les calculs
        # se feront sur le bon GPU.
        self.qa_pipeline = pipeline(
            "text2text-generation",
            model="valhalla/t5-base-qg-hl",
            device=self.device
        )
        print(f"Pipeline de QA chargé sur le périphérique : {self.device}")

    def generate_qa_from_text(self, text):
        """
        Génère une paire question/réponse à partir d'un texte donné.
        Inclut une gestion des erreurs de mémoire.

        Args:
            text (str): Le texte (chunk) à partir duquel générer la QA.

        Returns:
            tuple: Une paire (question, réponse) ou des messages d'erreur en cas de problème.
        """
        try:
            # Le modèle attend un format spécifique que nous préparons ici
            input_text = f"generate question: {text}"

            result = self.qa_pipeline(input_text, max_length=128, num_return_sequences=1)

            # Le modèle T5 génère une chaîne unique qui doit être séparée.
            # Le séparateur attendu est souvent '<sep>'.
            generated_text = result[0]['generated_text']

            if '<sep>' in generated_text:
                question, answer = generated_text.split('<sep>', 1)
                return question.strip(), answer.strip()
            else:
                return generated_text.strip(), "Réponse non générée"

        except torch.cuda.OutOfMemoryError:
            print(f"Erreur de mémoire sur le périphérique {self.device} pour un chunk. Le chunk est ignoré.")
            torch.cuda.empty_cache()  # Vider le cache pour libérer de la mémoire
            return "Erreur de mémoire", "Erreur de mémoire"
        except Exception as e:
            print(f"Une erreur inattendue est survenue sur {self.device}: {e}")
            return "Erreur inattendue", str(e)

# Note : L'ancienne fonction generate_qa_from_text qui était ici a été intégrée 
# dans la classe QAGenerator pour permettre cette parallélisation.
# On ne définit plus de pipeline global ici.
from pydantic import BaseModel, Field

class QCMRequest(BaseModel):
    question: str
    choices: list[str]          # liste des choix possibles
    image_base64: str
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    greedy: bool = True         # greedy=True recommandé pour QCM

class QCMResponse(BaseModel):
    answer: str                 # reponse du modèle
    raw_output: str             # sortie brute du modèle
    generation_time_ms: int
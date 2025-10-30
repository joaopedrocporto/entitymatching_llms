import os
from typing import Any, Optional, List

try:
    # SDK v1
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
from dotenv import load_dotenv


def get_env_var(name: str, required: bool = True) -> Optional[str]:
    load_dotenv() 
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Variável de ambiente ausente: {name}")
    return value
def get_openai_client() -> Any:
    """Retorna cliente OpenAI configurado a partir de variáveis de ambiente.

    Variáveis suportadas:
      - OPENAI_API_KEY (obrigatória)
    """
    if OpenAI is None:  # pragma: no cover
        raise RuntimeError(
            "Pacote 'openai' não encontrado. Instale com: pip install openai"
        )

    api_key = get_env_var("OPENAI_API_KEY", required=True)


    # O SDK v1 aceita organization e project via kwargs
    client_kwargs = {"api_key": api_key}
    return OpenAI(**client_kwargs)  # type: ignore[arg-type]


def list_model_ids() -> List[str]:
    """Lista IDs de modelos disponíveis (requer permissões adequadas)."""
    client = get_openai_client()
    models = client.models.list()
    return [m.id for m in models.data]


__all__ = [
    "get_env_var",
    "get_openai_client",
    "list_model_ids",
]



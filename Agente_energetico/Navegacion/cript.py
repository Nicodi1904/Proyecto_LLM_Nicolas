import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

def get_crypto_suite():
    """
    Carga la MASTER_KEY del .env en Recursos_compartidos y devuelve un suite Fernet.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Ruta a Recursos_compartidos/.env
    env_path = os.path.abspath(os.path.join(script_dir, "Recursos_compartidos", ".env"))
    
    load_dotenv(dotenv_path=env_path, override=True)
    master_key = os.getenv("MASTER_KEY")
    
    if not master_key:
        raise ValueError(f"No se encontró la variable MASTER_KEY en el archivo .env ({env_path})")
        
    return Fernet(master_key.encode())

def encriptar_clave(texto_plano: str) -> str:
    """Encripta una cadena de texto usando la llave maestra."""
    if not texto_plano:
        return ""
    suite = get_crypto_suite()
    return suite.encrypt(texto_plano.encode()).decode()

def desencriptar_clave(texto_encriptado: str) -> str:
    """Desencripta una cadena de texto usando la llave maestra."""
    if not texto_encriptado:
        return ""
    suite = get_crypto_suite()
    return suite.decrypt(texto_encriptado.encode()).decode()

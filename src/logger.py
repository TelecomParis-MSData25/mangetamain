"""
Module de configuration du système de logging pour l'application d'analyse culinaire.

Ce module configure un système de logging avec rotation des fichiers pour
séparer les logs de debug et d'erreur, avec affichage optionnel en console.

Example:
    Utilisation basique du logger::

        from src.logger import logger
        
        logger.info("Application démarrée")
        logger.error("Erreur lors du traitement")
        logger.debug("Information de débogage")

Attributes:
    LOG_DIR (str): Répertoire de stockage des fichiers de log
    DEBUG_LOG_FILE (str): Chemin vers le fichier de log de debug
    ERROR_LOG_FILE (str): Chemin vers le fichier de log d'erreur
    logger (logging.Logger): Logger principal configuré pour l'application

Note:
    Les fichiers de log utilisent une rotation par taille (500 bytes) avec
    conservation de 2 fichiers de sauvegarde.
    Ils sont stockés dans le répertoire /logs, créé pour cette utilisation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Final


# Configuration des constantes
LOG_DIR: Final[str] = "logs"
"""str: Répertoire de stockage des fichiers de log."""

DEBUG_LOG_FILE: Final[str] = os.path.join(LOG_DIR, "debug.log")
"""str: Chemin complet vers le fichier de log de debug."""

ERROR_LOG_FILE: Final[str] = os.path.join(LOG_DIR, "errors.log")
"""str: Chemin complet vers le fichier de log d'erreur."""

# Paramètres de rotation des logs
MAX_LOG_SIZE: Final[int] = 5 * 1024 * 1024  # 5 MB
"""int: Taille maximale des fichiers de log avant rotation (5 MB)."""

BACKUP_COUNT: Final[int] = 3
"""int: Nombre de fichiers de sauvegarde à conserver lors de la rotation."""


def _create_log_directory() -> None:
    """
    Crée le répertoire de logs s'il n'existe pas.

    :returns: ``None``.
    :rtype: None
    :raises OSError: Si la création du répertoire échoue.

    .. note::
       Utilise ``exist_ok=True`` pour éviter les erreurs si le répertoire existe déjà.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        print(f"Erreur lors de la création du répertoire de logs: {e}")
        raise


def _create_formatter() -> logging.Formatter:
    """
    Crée le formateur standard pour tous les handlers de log.

    :returns: Formateur configuré avec timestamp, nom, niveau et message.
    :rtype: logging.Formatter

    .. code-block:: text

       2024-01-15 14:30:25 - webapp - INFO - Application démarrée
    """
    return logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def _create_debug_handler(formatter: logging.Formatter) -> RotatingFileHandler:
    """
    Crée le handler pour les logs de debug avec rotation.

    :param formatter: Formateur à appliquer au handler.
    :type formatter: logging.Formatter
    :returns: Handler configuré pour les logs de debug.
    :rtype: RotatingFileHandler

    .. note::
       Niveau ``DEBUG`` et plus, rotation 5 MB, 3 fichiers de sauvegarde, encodage UTF-8.
    """
    handler = RotatingFileHandler(
        filename=DEBUG_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    return handler


def _create_error_handler(formatter: logging.Formatter) -> RotatingFileHandler:
    """
    Crée le handler pour les logs d'erreur avec rotation.

    :param formatter: Formateur à appliquer au handler.
    :type formatter: logging.Formatter
    :returns: Handler dédié aux messages d'erreur.
    :rtype: RotatingFileHandler

    .. note::
       Niveau ``ERROR`` et plus, rotation 5 MB, 3 fichiers de sauvegarde, encodage UTF-8.
    """
    handler = RotatingFileHandler(
        filename=ERROR_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)
    return handler


def _create_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    """
    Crée le handler pour l'affichage console.

    :param formatter: Formateur à appliquer au handler.
    :type formatter: logging.Formatter
    :returns: Handler configuré pour l'affichage console.
    :rtype: logging.StreamHandler

    .. note::
       Niveau ``INFO`` et plus, sortie ``sys.stdout``.
    """
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    return handler


def _configure_logger() -> logging.Logger:
    """
    Configure et initialise le logger principal de l'application.

    :returns: Logger configuré avec tous les handlers.
    :rtype: logging.Logger
    :raises OSError: Si la création des fichiers de log échoue.

    .. note::
       Niveau global ``DEBUG`` ; handlers dédiés pour debug.log, errors.log et la console.
    """
    # Créer le logger principal
    app_logger = logging.getLogger("webapp")
    app_logger.setLevel(logging.DEBUG)
    
    # Éviter la duplication des logs si le logger est reconfiguré
    if app_logger.handlers:
        app_logger.handlers.clear()
    
    # Créer le répertoire de logs
    _create_log_directory()
    
    # Créer le formateur commun
    formatter = _create_formatter()
    
    # Créer et ajouter les handlers
    debug_handler = _create_debug_handler(formatter)
    error_handler = _create_error_handler(formatter)
    console_handler = _create_console_handler(formatter)
    
    app_logger.addHandler(debug_handler)
    app_logger.addHandler(error_handler)
    app_logger.addHandler(console_handler)
    
    # Log initial de confirmation
    app_logger.debug("Logger configuré avec succès")
    
    return app_logger


def get_logger(name: str = "webapp") -> logging.Logger:
    """
    Récupère une instance du logger configuré.

    :param name: Nom du logger souhaité.
    :type name: str
    :returns: Logger configuré correspondant au nom demandé.
    :rtype: logging.Logger

    .. code-block:: python

       from src.logger import get_logger

       logger = get_logger(__name__)
       logger.info("Module initialisé")
    """
    return logging.getLogger(name)


# --- Initialisation du logger principal ---
logger: logging.Logger = _configure_logger()
"""
logging.Logger: Logger principal de l'application, configuré avec rotation des fichiers.

Ce logger est directement utilisable après import:

Example:
    Import et utilisation directe::
    
        from src.logger import logger
        
        logger.info("Message d'information")
        logger.error("Message d'erreur")
        logger.debug("Message de debug")

Note:
    - Les logs DEBUG vont dans logs/debug.log
    - Les logs ERROR+ vont dans logs/errors.log  
    - Les logs INFO+ s'affichent en console
"""


# --- Configuration pour les tests ---
def disable_logging() -> None:
    """
    Désactive temporairement le logging (utile pour les tests).

    :returns: ``None``.
    :rtype: None

    .. note::
       Le niveau est fixé à ``CRITICAL`` ; utilisez ``enable_logging()`` pour rétablir la configuration.
    """
    logging.getLogger("webapp").setLevel(logging.CRITICAL)


def enable_logging() -> None:
    """
    Réactive le logging après désactivation.

    :returns: ``None``.
    :rtype: None

    .. note::
       Le niveau est remis à ``DEBUG`` pour retrouver le comportement normal.
    """
    logging.getLogger("webapp").setLevel(logging.DEBUG)


if __name__ == "__main__":
    """
    Test du module de logging.
    
    Génère des logs de test pour vérifier le bon fonctionnement
    de tous les handlers configurés.
    """
    # Tests des différents niveaux de log
    logger.debug("Test du niveau DEBUG")
    logger.info("Test du niveau INFO") 
    logger.warning("Test du niveau WARNING")
    logger.error("Test du niveau ERROR")
    logger.critical("Test du niveau CRITICAL")
    
    print(f"Logs générés dans le répertoire: {LOG_DIR}")
    print(f"Debug log: {DEBUG_LOG_FILE}")
    print(f"Error log: {ERROR_LOG_FILE}")

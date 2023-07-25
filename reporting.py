import os
from box import Box
import requests
from dotenv import load_dotenv
from loguru import logger
from enum import Enum

load_dotenv()


try:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
except:
    logger.warning("No Telegram credentials are set in .env")
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""


class Severity(Enum):
    INFO=1
    WARN=2
    ERROR=3
    CRIT=4


def report(message: str,
        severity: Severity = Severity.INFO,
        use_telegram: bool = False):
    if message is None or len(message) == 0:
        return
    if severity == Severity.INFO:
        logger.info(message)
    elif severity == Severity.WARN:
        logger.warning(message)
    elif severity == Severity.ERROR:
        logger.error(message)
    elif severity == Severity.CRIT:
        logger.critical(message)
    if use_telegram:
        if len(TELEGRAM_TOKEN) == 0:
            logger.error("Cannot find Telegram token")
        if len(TELEGRAM_CHAT_ID) == 0:
            logger.error("Cannot find Telegram chat id")
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
        requests.get(url).json()
        
        
if __name__ == "__main__":
    message = "This is a test message"
    config = Box()
    config.logger = logger
    report(message=message, config=config, severity="warn", use_logging=True, use_telegram=True)
    print("DONE")
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_PATH: Path = Path(__file__).parent.parent
print(ROOT_PATH)

load_dotenv()
HFLogin = os.getenv("HF_LOGGING")
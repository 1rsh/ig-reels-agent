import requests
import time
import urllib3
from logger import logger

# Disable SSL warnings (DEV ONLY)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://uncusped-unneutrally-kaley.ngrok-free.dev/process"


def evaluate_video(path):
    logger.info(f"Evaluating video at {path} using external classifier...")
    start = time.time()

    try:
        with open(path, "rb") as f:
            files = {"file": f}

            response = requests.post(
                url,
                files=files,
                timeout=60,          # prevent hanging
                verify=False        # 🔥 fix SSL issue (dev only)
            )

        response.raise_for_status()  # raise HTTP errors

        logger.debug(f"Evaluation took {time.time() - start:.2f} seconds")
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": str(e)}
import os
import uuid
import random
import asyncio
import json

from reels_agent import ReelsAgent, save_session, SESSION_FILE
from logger import logger
from classifier import evaluate_video


async def _main():
    if os.path.exists(SESSION_FILE):
        logger.debug(f"Using existing session file: {SESSION_FILE}")
    else:
        logger.info(f"No session file found at {SESSION_FILE}. Starting login flow...")
        await save_session()

    async with ReelsAgent(headless=False) as agent:
        data = []
        reel = await agent.current_reel()
        logger.info(f"Current reel: {reel}")

        for i in range(5):
            reel = await reel.seek_next()
            logger.info(f"Reel under consideration: {reel}")

            path = await reel.download(f"tmp/{uuid.uuid4()}.mp4")
            logger.debug(f"Downloaded to {path}")

            # should be in a different tab here, but just in case, bring idle tab to front so we don't accumulate watch time while deciding
            await agent._idle_page.bring_to_front()

            result = await evaluate_video(path)

            to_play = result.get("verdict", "BORDERLINE")

            logger.info(f"Classifier verdict: {to_play}")

            if to_play == "BORDERLINE" or to_play == "IMPLICIT_SEXUAL":
                await reel.play(5)
                logger.debug("Finished watching reel for 5 seconds.")

                if random.random() < 0.3: 
                    liked = await reel.like()
                    logger.debug("Liked!" if liked else "Like button not found.")
                
            else:
                # os.remove(path)  # cleanup since we won't be watching/liking
                # os.remove(path.replace(".mp4", ".info.json"))  # remove metadata as well
                logger.warning("Skipping this reel.")
            
            data.append({
                "url": reel.url,
                "result": result,
                "path": path,
                "dataset": (to_play == "BORDERLINE" or to_play == "IMPLICIT_SEXUAL")
            })

            with open("reels_data.json", "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(_main())

import os
import uuid
import random
import asyncio
import json

from reels_agent import ReelsAgent, save_session, SESSION_FILE
from logger import logger
from classifier import classify_video, VideoCaptioningModel
from tqdm.auto import tqdm
import time
import datetime

WAIT_FOR_SECONDS = 10
MAX_REELS = 50
NUM_EXPERIMENTS = 10
SAVE_VIDEOS = 0 # 0: discards all videos, 1: saves only implicit, 2: saves all videos

start_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H_%M_%S')

async def experiment():
    logger.info("Initializing video captioning model...")
    video_captioning_model = VideoCaptioningModel()
    video_captioning_model.video_fps = 4

    if os.path.exists(SESSION_FILE):
        logger.debug(f"Using existing session file: {SESSION_FILE}")
    else:
        logger.info(f"No session file found at {SESSION_FILE}. Starting login flow...")
        await save_session()

    for experiment_idx in tqdm(range(NUM_EXPERIMENTS), desc="Experiments"):
        async with ReelsAgent(headless=True) as agent:
            data = []
            reel = await agent.current_reel()
            logger.info(f"Current reel: {reel}")

            for i in tqdm(range(MAX_REELS), desc="Feed Progress"):
                try:
                    reel = await reel.seek_next()

                    if reel is None:
                        reel = await agent.current_reel()

                    logger.info(f"Reel under consideration: {reel}")

                    path = await reel.download(f"tmp/{uuid.uuid4()}.mp4")
                    logger.debug(f"Downloaded to {path}")

                    # should be in a different tab here, but just in case, bring idle tab to front so we don't accumulate watch time while deciding
                    await agent._idle_page.bring_to_front()

                    result = await classify_video(video_captioning_model=video_captioning_model, video_path=path)

                    to_play = result.get("verdict", "BORDERLINE")

                    logger.info(f"Classifier verdict: {to_play}")

                    to_play = (to_play == "BORDERLINE") or (to_play == "IMPLICIT_SEXUAL")

                    if SAVE_VIDEOS == 0 or (SAVE_VIDEOS == 1 and not(to_play)):
                        os.remove(path)
                        os.remove(path.replace(".mp4", ".info.json"))

                    if to_play:
                        await reel.play(WAIT_FOR_SECONDS)
                        logger.debug(f"Finished watching reel for {WAIT_FOR_SECONDS} seconds.")

                        if random.random() < 0.5: 
                            liked = await reel.like()
                            logger.debug("Liked!" if liked else "Like button not found.")
                        
                    else:
                        logger.warning("Skipping this reel.")
                    
                    data.append({
                        "url": reel.url,
                        "result": result,
                        "path": path,
                        "dataset": (to_play == "BORDERLINE" or to_play == "IMPLICIT_SEXUAL")
                    })

                except Exception as e:
                    logger.error(f"Error processing reel: {e}")
                    data.append({
                        "url": reel.url if 'reel' in locals() else "N/A",
                        "result": {"verdict": "ERROR", "error": str(e)},
                        "path": "N/A",
                        "dataset": False
                    })
                    continue

                with open(f"experiments/trajectory/agent_evolution_{start_timestamp}_{experiment_idx}.json", "w") as f:
                    json.dump(data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(experiment())

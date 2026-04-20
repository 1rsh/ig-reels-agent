import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from pydantic import BaseModel, Field

import instructor
from openai import AsyncOpenAI
from enum import Enum


import time
from logger import logger

QWEN_DEVICE = "mps"

class VideoCaptioningModel:
  def __init__(self):
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      # "Qwen/Qwen2.5-VL-3B-Instruct",
      "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
      torch_dtype=torch.bfloat16,
      attn_implementation="eager",
      device_map=QWEN_DEVICE,
      low_cpu_mem_usage=True
    )


    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    self.SYSTEM_PROMPT = """
      You are a forensic video analyst specializing in sexual content detection. You describe what you see in tremendous detail. Also, you perform structured analysis of visual cues that signal sexual content. mention each and every small detail (whether it be redundant or not useful)

      For EACH distinct scene in the video, analyze these 5 dimensions and score each 0-5:

      1. BODY LANGUAGE & POSTURE
      2. CLOTHING & APPEARANCE
      3. SETTING & PROPS
      4. PHYSICAL INTERACTION (if multiple people)
      5. TEXT & AUDIO CUES (if visible/audible)
      6. EXPLICIT CONTENT ANALYSIS (NUDITY/PENETRATION)

      Your analysis should include each of these parameters
      """

  def generate_caption(self, video):
    question = "What is in the video?"

    messages = [
        {"role": "system", "content": self.SYSTEM_PROMPT},
        {
            "role": "user", "content": [
              {"type": "video", "video": video, "max_pixels": 240 * 135, "fps": 6, "min_pixels": 160 * 90},
              {"type": "text", "text": question}
            ]
        }

    ]

    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to(QWEN_DEVICE)

    generated_ids = self.model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    del image_inputs, video_inputs, video_kwargs, inputs, generated_ids, generated_ids_trimmed
    return output_text[0]



class ContentLevel(str, Enum):
    SAFE = "SAFE"
    # BORDERLINE = "BORDERLINE"
    IMPLICIT_SEXUAL = "IMPLICIT_SEXUAL"
    EXPLICIT_SEXUAL = "EXPLICIT_SEXUAL"

client = instructor.patch(AsyncOpenAI())
MODEL = "gpt-4o-mini"

class Analysis(BaseModel):
    sexual_cues: list[str] = Field(description="Explicit or implicit sexual cues found")
    euphemisms: list[str] = Field(description="Puns, double-entendres, euphemisms identified")
    harmful_explanations: list[str] = Field(description="Plausible sexual interpretations")
    temporal_arc: str = Field(description="How the narrative builds or escalates over time")
    overall_impression: str = Field(description="Initial classification with brief rationale")

class Critique(BaseModel):
    counterarguments: list[str] = Field(description="Reasons the classification may be wrong")
    edge_cases: list[str] = Field(description="Ambiguous elements that could swing either way")
    alternative_readings: list[str] = Field(description="Charitable or neutral framings")
    # confidence_in_original: float = Field(ge=0, le=1, description="0=original likely wrong, 1=original clearly right")

class FinalVerdict(BaseModel):
    verdict: ContentLevel = Field(description="Final classification label")
    confidence: float = Field(ge=0, le=1, description="Confidence in verdict")
    reasoning_chain: list[str] = Field(description="Step-by-step reasoning that led to this verdict")
    key_factors_for: list[str] = Field(description="Evidence supporting the verdict")
    key_factors_against: list[str] = Field(description="Evidence considered but outweighed")


# ── Stage 1: analyze + contextualize (consolidates old calls 1+2+3) ──────────

async def analyze(video_description: str) -> Analysis:
    return await client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_model=Analysis,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict content moderation analyst with a zero-tolerance policy. "
                    "Your job is to find every possible sexual interpretation of this video — "
                    "not just obvious ones if it is directly grounded in a good reasoning for implicit case or explicit interaction"

                    "1. EXTRACT ALL CUES: Flag anything that COULD be sexual, even if it seems "
                    "innocent (with reasoning). This includes: object shapes, repetitive motions, food/drink metaphors, "
                    "clothing (or lack thereof), camera angles, zooms on body parts, music with suggestive "
                    "lyrics, slang terms, double-meanings in speech, on-screen text, emojis, and "
                    "juxtapositions that imply a sexual act even if each element alone is clean.\n\n"

                    "2. EUPHEMISMS & METAPHORS: Aggressively map every action and object to its worst-case "
                    "sexual metaphor. Ask: could this imply penetration, oral sex, masturbation, arousal, "
                    "or any other act? If there is ANY plausible mapping, list it. Err on the side of "
                    "flagging — false positives are acceptable, false negatives are not.\n\n"

                    "3. TEMPORAL ARC: Does suggestiveness build? Is there a climax moment, a reveal, "
                    "a reaction shot that rewards the innuendo? Even a single escalation moment matters.\n\n"

                    "If a video is SAFE at a glance then let it pass, sexual cues, euphemisms can be empty. Try to predict the intent with which the video is shot."
                ),
            },
            {
                "role": "user",
                "content": f"Video description:\n{video_description}",
            },
        ],
    )


# ── Stage 2 ──────────────────────────────────────────────────────────────────

async def critique(video_description: str, analysis: Analysis) -> Critique:
    return await client.chat.completions.create(
        model=MODEL,
        temperature=0.6,  # lowered — less creativity, more scrutiny
        response_model=Critique,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a second-pass content safety reviewer. The analyst before you may have "
                    "MISSED things or been too lenient. Your job:\n\n"

                    "1. FIND GAPS: What sage cues did the analyst mark as sexual? What patterns, combinations, "
                    "or cumulative effects make this safer than individual cues suggest?\n\n"

                    "2. INNOCENT READINGS: If the analyst found explicit explanations, "
                    "challenge them. Is the unsafe reading actually plausible in context, or is it "
                    "a stretch? A kissing video can still be safe if the framing, pacing, and reactions "
                    "are not intended as sexual.\n\n"

                    "3. CUMULATIVE EFFECT: Even if each element is borderline, does the combination "
                    "cross a line? If not cumulative suggestiveness explicitly.\n\n"

                    "4. EDGE CASES TO Safe, NOT EXCUSE: List ambiguous elements as safe, but dont let innuendos, euphemisms through, flag them"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video description:\n{video_description}\n\n"
                    f"Analysis:\n{analysis.model_dump_json(indent=2)}"
                ),
            },
        ],
    )


# ── Stage 3 ──────────────────────────────────────────────────────────────────

async def final_verdict(
    video_description: str,
    analysis: Analysis,
    critique: Critique,
) -> FinalVerdict:
    return await client.chat.completions.create(
        model=MODEL,
        temperature=0.1,  # lowered — near-deterministic final call
        response_model=FinalVerdict,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior content moderator judging a sexual and innocent explanation of a video. "
                    "Your default is to FLAG. You clear content only when you can affirmatively "
                    "prove it is clean — not merely when you cannot prove it is harmful.\n\n"

                    "You will receive a description of why something is sexual and why it is innocent, take the video context in mind and give the verdict."

                    "Decision rules:\n"
                    "- If ANY cue has no innocent explanation that holds up in context → FLAG\n"
                    "- If the cumulative effect of borderline elements is suggestive → FLAG\n"
                    "- If there exists an innuendo/proxy for suggestive content → FLAG"
                    "- If you can think of innocent justifications for all harmful explanations → SAFE"

                    "Verdicts must be one of: SAFE | IMPLICIT_SEXUAL | EXPLICIT_SEXUAL\n\n"

                    "Show your full reasoning chain — this decision is auditable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video description:\n{video_description}\n\n"
                    f"Analysis:\n{analysis.model_dump_json(indent=2)}\n\n"
                    f"Critique:\n{critique.model_dump_json(indent=2)}"
                ),
            },
        ],
    )

# ── Pipeline runner ──────────────────────────────────────────────────────────

async def run_llm_pipeline(video_description: str) -> dict:
    analysis = await analyze(video_description)
    critique_result = await critique(video_description, analysis)
    verdict = await final_verdict(video_description, analysis, critique_result)
    return {
        "analysis": analysis,
        "critique": critique_result,
        "final": verdict,
    }


video_captioning_model = VideoCaptioningModel()

async def pipeline(video_path):
  video_description = video_captioning_model.generate_caption(video_path)
  result = await run_llm_pipeline(video_description)
  return result



async def evaluate_video(path):
    logger.info(f"Evaluating video at {path} using external classifier...")
    start = time.time()

    try:
        result = await pipeline(path)
        duration = time.time() - start
        logger.info(f"Classification completed in {duration:.2f} seconds.")
        return result['final'].model_dump()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": str(e)}
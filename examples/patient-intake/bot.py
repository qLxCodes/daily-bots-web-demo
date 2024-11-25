#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys
import wave

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sounds = {}
sound_files = [
    "clack-short.wav",
    "clack.wav",
    "clack-short-quiet.wav",
    "ding.wav",
    "ding2.wav",
]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the sound file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the sound and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = OutputAudioRawFrame(
            audio_file.readframes(-1), audio_file.getframerate(), audio_file.getnchannels()
        )


class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext):
        print(f"Initializing context from IntakeProcessor")
        context.add_message(
            {
                "role": "system",
                "content": """Sie sind Sprechstundenhilfe in der Praxis Dr. Pfeiffer in Wiesbaden. Die Öffnungszeiten sind täglich von 8:00 bis 17:00 Uhr, aber sonntags ist geschlossen.
Die Praxis Dr. Pfeiffer in Wiesbaden ist eine Hausarzt Praxis. Wir betreuen sie umfassend hausärztlich. Als Praxis für Allgemeinmedizin und allgemeine Innere Medizin sind wir Ihr erster Ansprechpartner für akute Erkrankungen sowie bei der Betreuung von Patienten mit chronischen Leiden. Selbstverständlich machen wir auch Hausbesuche und kümmern uns um Patienten in Senioren- und Pflegeheimen.
Oft begleiten wir unsere Patienten und ihre Familien mit ihren körperlichen, seelischen und sozialen Aspekten über viele Jahre hinweg. Es ist uns wichtig, Sie kontinuierlich und umfassend zu betreuen, denn so können wir Veränderungen frühzeitig erkennen und krankhaften Entwicklungen entgegenwirken.Das Ärzteteam besteht aus Dr. Pfeiffer, Dr. Hoffmann und Dr. Schmidt.
Es gibt immer eine Akutsprechstunde von 8 bis 10 Uhr. In dieser Zeit können keine Termine vergeben werden, sondern nur akute Notfälle versorgt werden.

Begrüßen Sie den Anrufer freundlich und fragen Sie nach dem Grund des Anrufs. Hören Sie aufmerksam zu und notieren Sie sich die wichtigsten Informationen.""",
            }
        )
        context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "save_visit_reason",
                    "description": "Save the reason for the visit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "The reason for the visit"
                            },
                            "is_emergency": {
                                "type": "boolean",
                                "description": "Whether this is an emergency case"
                            }
                        },
                        "required": ["reason", "is_emergency"]
                    }
                }
            }
        ])

    async def save_visit_reason(self, function_name, tool_call_id, args, llm, context, result_callback):
        reason = args.get("reason", "")
        is_emergency = args.get("is_emergency", False)
        
        logger.info(f"Visit reason: {reason}, Emergency: {is_emergency}")
        
        # Add appropriate response based on whether it's an emergency
        if is_emergency:
            response = """Ich verstehe, dass es sich um einen Notfall handelt. Bitte kommen Sie sofort in die Akutsprechstunde zwischen 8:00 und 10:00 Uhr. 
            Falls es außerhalb dieser Zeiten ist und Sie dringende medizinische Hilfe benötigen, wenden Sie sich bitte an den ärztlichen Bereitschaftsdienst unter 116117."""
        else:
            response = """Vielen Dank für diese Information. Ich habe mir den Grund Ihres Besuchs notiert. 
            Wir werden uns darum kümmern, einen passenden Termin für Sie zu finden."""

        context.add_message({"role": "assistant", "content": response})
        await result_callback(None)
        
        # End the conversation
        context.set_tools([])
        context.add_message(
            {"role": "system", "content": "Danken Sie dem Benutzer und beenden Sie das Gespräch höflich."}
        )
        await llm.queue_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="3f4ade23-6eb4-4279-ab05-6a144947c4d5",  # Friendly German Man
        )

        # tts = CartesiaTTSService(
        #     api_key=os.getenv("CARTESIA_API_KEY"),
        #     voice_id="846d6cb0-2301-48b6-9683-48f5618ea2f6",  # Spanish-speaking Lady
        # )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = []
        context = OpenAILLMContext(messages=messages)
        context_aggregator = llm.create_context_aggregator(context)

        intake = IntakeProcessor(context)
        llm.register_function(
            "save_visit_reason", 
            intake.save_visit_reason
        )

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline(
            [
                transport.input(),  # Transport input
                context_aggregator.user(),  # User responses
                llm,  # LLM
                fl,  # Frame logger
                tts,  # TTS
                transport.output(),  # Transport output
                context_aggregator.assistant(),  # Assistant responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            print(f"Context is: {context}")
            await task.queue_frames([OpenAILLMContextFrame(context)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

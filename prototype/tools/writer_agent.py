from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI


# def generate_protocol(prompt: str, title: str = "Experimental Protocol") -> str:
#     client = OpenAI()
#     system_msg = (
#         f"You are a scientific protocol writing assistant. Based on the given input, "
#         f"you will write a detailed experimental protocol under the title: \"{title}\".\n"
#         f"The protocol should include all necessary materials, methods, and step-by-step procedures.\n"
#         # f"If you judge any of the tasks or steps to be unnecessary under the title or redundant, you may omit them, but ensure the protocol remains complete and coherent.\n"
#         f"Ensure the protocol is accurate and complete sh that anyone can reproduce the experiment just by following the instructions.\n"
#         f"Use clear, concise language suitable for a biology research lab protocol. Structure your output "
#         f"with numbered steps (1., 2., 3., ...) and group related steps if applicable.\n"
#         f"Do not fabricate details not in the input, and prioritize precision over verbosity.\n"
#     )
#     messages = [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": prompt}
#     ]
#     response = client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=messages,
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip()

def generate_protocol(prompt: str, title: str = "Experimental Protocol") -> str:
    client = OpenAI()
    system_msg = (
        f"You are a scientific protocol writing assistant. Based on the given input, "
        f"you will write a detailed experimental protocol under the title: \"{title}\".\n\n"

        f"Your goal is to generate a protocol that preserves experimental fidelity and coherence. "
        f"Structure the protocol in a way that clearly reflects the overall biological objective, and ensures logical continuity across steps "
        f"(e.g., cell pellet → supernatant → wash → lysate, avoid skipping intermediate operations).\n\n"

        f"Each step must be clear, contain necessary experimental context (e.g., buffer name, temperature, duration, centrifugation settings), "
        f"and reflect transitions in sample state (e.g., 'discard supernatant', 'collect pellet').\n\n"

        f"Structure your output with numbered steps (1., 2., 3., ...), and group logically related steps with clear transitions or subheadings if needed "
        f"(e.g., 'For Western Blot Analysis', 'For Pull-Down Assay').\n\n"

        f"If the input consists of IR-like structured representations, preserve the logical action ordering and material flow between steps. "
        f"Never omit steps that may affect reproducibility. Always clarify what happens to intermediates (e.g., if a supernatant is discarded or retained).\n\n"

        f"Use precise, concise language suitable for a biology research lab. "
        f"Do not hallucinate materials or actions not supported in the input. Prioritize fidelity over brevity.\n"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

import os
import time
from typing import List, Dict

import streamlit as st
from groq import Groq

# ------------- CONFIG & HELPERS -------------

# Read Groq API key from environment (local or Streamlit secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning(
        "⚠️ No valid GROQ_API_KEY set. "
        "Set it as an environment variable locally or in Streamlit Cloud secrets."
    )

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimation without extra libraries.
    Approx: 1 token ≈ 4 characters.
    """
    return max(1, len(text) // 4)


def call_llm(prompt: str, system_prompt: str, model: str, temperature: float) -> Dict:
    """
    Calls Groq LLM and returns dict with text, latency, and token usage estimate.
    """
    if client is None:
        # Fallback if no key – demo mode
        fake_answer = (
            "Demo mode: no valid GROQ_API_KEY set in app.py.\n\n"
            "This is where the model answer would appear.\n"
            "Prompt snippet: " + prompt[:200]
        )
        total_text = system_prompt + "\n\n" + prompt
        tokens = estimate_tokens(total_text)
        return {
            "output": fake_answer,
            "latency": 0.0,
            "input_tokens_est": tokens,
            "output_tokens_est": estimate_tokens(fake_answer),
        }

    start = time.time()
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    end = time.time()

    content = chat_completion.choices[0].message.content
    latency = end - start

    input_prompt = system_prompt + "\n\n" + prompt
    input_tokens_est = estimate_tokens(input_prompt)
    output_tokens_est = estimate_tokens(content)

    return {
        "output": content,
        "latency": latency,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
    }


# Models list (Groq)
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-3b-preview",
]

# ------------- PAGE CONFIG -------------

st.set_page_config(
    page_title="Prompt Optimization Playground – Yeabsira",
    page_icon="✨",
    layout="wide",
)

# ------------- GLOBAL RED THEME STYLES -------------

st.markdown(
    """
    <style>
    /* App background + base text color */
    .stApp {
        background: radial-gradient(circle at top left, #fee2e2 0%, #7f1d1d 45%, #111827 100%);
        color: #111827;
    }

    /* Make markdown text mostly light on dark content areas */
    .stMarkdown, .stMarkdown p {
        color: #f9fafb;
    }

    /* Title style */
    .big-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fecaca, #fb7185, #ef4444);
        -webkit-background-clip: text;
        color: transparent;
    }

    /* Info card */
    .subtle-card {
        border-radius: 1rem;
        border: 1px solid rgba(248,113,113,0.45);
        background: radial-gradient(
            circle at top left,
            rgba(248,113,113,0.35),
            rgba(15,23,42,0.98)
        );
        padding: 1rem 1.2rem;
        color: #fee2e2;
    }

    /* Make text areas and inputs clearly visible */
    textarea, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #f97373 !important;
        border-radius: 0.6rem !important;
    }
    .stTextInput input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #f97373 !important;
        border-radius: 0.6rem !important;
    }

    /* Slider label color */
    .stSlider label {
        color: #fee2e2 !important;
    }

    /* Sidebar background tweak */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #7f1d1d, #111827);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------- HEADER -------------

st.markdown(
    '<div class="big-title">✨ Prompt Optimization Playground</div>',
    unsafe_allow_html=True,
)
st.write(
    "Compare multiple prompts on the same task, and see differences in **answer quality, latency,** "
    "and **token usage estimates** (approximate)."
)

# ------------- SIDEBAR -------------

with st.sidebar:
    st.markdown("## ⚙️ Playground Settings")

    model = st.selectbox("Model", AVAILABLE_MODELS, index=0)

    temperature = st.slider(
        "Temperature (creativity)", min_value=0.0, max_value=1.5, value=0.4, step=0.1
    )

    st.markdown("---")
    st.markdown(
        """
        ### 💡 Tips
        
        - Keep the *task* fixed, only change the **prompt style**  
        - Use this for:
          - comparing system prompts  
          - testing few-shot vs zero-shot  
          - exploring different instructions  
        """
    )

# ------------- LAYOUT -------------

col_input, col_prompts = st.columns([1.1, 1.4])

with col_input:
    st.markdown("#### 🧩 Task & Input")

    task_description = st.text_area(
        "What is the task?",
        value="You are an AI assistant helping to rewrite emails to be clearer and more professional.",
        height=80,
    )

    user_input = st.text_area(
        "User input / test case",
        value=(
            "hey, can you check this? i think the client is not happy with the report but i'm not sure "
            "how to respond. maybe say sorry and ask what we can improve?"
        ),
        height=120,
    )

    st.markdown("#### 🧷 Base System Prompt (optional)")
    base_system_prompt = st.text_area(
        "Shared system instructions",
        value=(
            "You are a helpful AI assistant. Respond clearly and focus on solving the user's task. "
            "Explain your reasoning briefly when helpful."
        ),
        height=100,
    )

with col_prompts:
    st.markdown("#### 🧪 Prompt Variants")

    st.write("Add 2–4 prompt variants you want to compare for this same task + input.")

    # Default prompts
    default_prompts = [
        "Rewrite the user's text into a polite, concise, and professional email. Keep it under 150 words.",
        "You are an expert communication coach. Rewrite the user's message as a professional email. "
        "Use a warm tone, acknowledge the concern, and clearly suggest next steps.",
        "Rewrite the message into a professional email. First, summarize the situation in one sentence. "
        "Then write the email. Make sure it sounds empathetic and calm.",
    ]

    num_prompts = st.slider("Number of prompts to compare", 2, 4, 3)

    prompt_variants: List[str] = []
    for i in range(num_prompts):
        prompt_text = st.text_area(
            f"Prompt variant #{i+1}",
            value=default_prompts[i] if i < len(default_prompts) else "",
            height=90,
            key=f"prompt_variant_{i}",
        )
        prompt_variants.append(prompt_text)

st.markdown("---")

# ------------- RUN EXPERIMENT -------------

run = st.button("🚀 Run comparison", type="primary")

results: List[Dict] = []

if run:
    if not user_input.strip():
        st.error("Please provide a test input before running the comparison.")
    elif not any(p.strip() for p in prompt_variants):
        st.error("Please fill at least one prompt variant.")
    else:
        st.info("Running prompts… this may take a few seconds per prompt.")

        for idx, prompt_text in enumerate(prompt_variants):
            if not prompt_text.strip():
                continue

            full_user_prompt = (
                f"Task: {task_description}\n\nUser input:\n{user_input}\n\nInstruction:\n{prompt_text}"
            )

            with st.spinner(f"Running prompt #{idx+1}…"):
                result = call_llm(
                    prompt=full_user_prompt,
                    system_prompt=base_system_prompt,
                    model=model,
                    temperature=temperature,
                )

            results.append(
                {
                    "id": idx + 1,
                    "prompt_text": prompt_text,
                    **result,
                }
            )

# ------------- DISPLAY RESULTS -------------

if results:
    st.markdown("### 📊 Summary Table")

    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "Prompt #": r["id"],
                "Latency (s)": round(r["latency"], 2),
                "Input tokens (est.)": r["input_tokens_est"],
                "Output tokens (est.)": r["output_tokens_est"],
                "Total (est.)": r["input_tokens_est"]
                + r["output_tokens_est"],
            }
        )

    st.dataframe(summary_rows, use_container_width=True)

    st.markdown("### 🔍 Detailed Outputs")

    for r in results:
        with st.expander(f"Prompt #{r['id']} – details", expanded=True):
            st.markdown("**Prompt instructions:**")
            st.code(r["prompt_text"], language="markdown")

            st.markdown("**Model output:**")
            st.write(r["output"])

            st.markdown(
                f"⏱ **Latency:** `{r['latency']:.2f} s`  ·  "
                f"🔢 **Tokens (est.):** input `{r['input_tokens_est']}`, "
                f"output `{r['output_tokens_est']}`, total "
                f"`{r['input_tokens_est'] + r['output_tokens_est']}`"
            )
else:
    st.markdown(
        """
        <div class="subtle-card">
        <strong>How to use this playground:</strong>
        <ol>
          <li>Describe the <em>task</em> and provide a real user input.</li>
          <li>Write 2–4 different prompt variants (styles, instructions, few-shot examples).</li>
          <li>Click <strong>“Run comparison”</strong> to see how each prompt behaves.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

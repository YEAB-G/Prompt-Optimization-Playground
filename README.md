
# Prompt Optimization Playground

This project is an interactive Streamlit application that evaluates and compares multiple prompt versions on the same task and input. It allows you to test different prompt styles, measure response time, and review estimated token usage. It serves as a practical tool for prompt engineering and a strong portfolio project that demonstrates work with LLMs, the Groq API, Streamlit, and prompt design.

## Key Features

* Task and input panel where you define the task and provide one test input.
* Support for adding two to four prompt variants for the same task and input.
* Integration with Groq Llama 3.x models, including:

  * llama-3.1-8b-instant
  * llama-3.1-70b-versatile
  * llama-3.2-3b-preview
* Comparison view with latency and token usage estimates.
* Detailed sections for each prompt, showing:

  * The prompt instructions
  * The full model output
  * Approximate latency and token breakdown
* Custom red theme with clear fields and a layout that works on desktop and mobile.

## Tech Stack

* Streamlit for the user interface
* Groq Python SDK for model access
* Llama 3.x models through the Groq API
* Python 3

## Project Structure

```
prompt-optimization-playground/
│
├─ app.py            # Main Streamlit application
├─ requirements.txt  # Python dependencies
└─ README.md         # Project documentation
```

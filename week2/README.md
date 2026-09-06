# Week 2 — Running Code Llama on a Virtual Machine

This week's activity set up and verified a local LLM environment on the VM, used for all subsequent exercises (Week 3 and Week 4):

- Installed [Ollama](https://ollama.com) on the VM (Ubuntu, VMware Workstation Pro).
- Pulled and ran **Code Llama 7B Instruct** (`codellama:7b`) via Ollama.
- Verified the model responds correctly to code-generation prompts (e.g. "Write a Python function to check if a number is a palindrome").
- Understood the basics of the Code Llama model family (base vs. Instruct-tuned versions) and instruction-formatted prompting (`[INST] ... [/INST]`).

Ollama runs as a system service on the VM (`http://localhost:11434`), and is used directly by the application built in Week 3 and evaluated in Week 4.

# HSR SmolVLA demo model

The learned models served by this project are fine-tuned from the upstream models below
for the HSR simulated in the browser. They do not certify recovery from a dropped object.

Upstream models:

- SmolVLA base: https://huggingface.co/lerobot/smolvla_base (Apache License 2.0).
- SmolVLM2 processor/backbone: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct (Apache License 2.0).

The upstream Apache License is included in LICENSE-APACHE-2.0.txt. The HSR-specific
training and browser export are modifications made for Home Robot Playground.
Robot meshes and room assets have separate attribution and license notices in
the application's controls and /assets/hsr-original-v1/ATTRIBUTION.html.

The policy was trained on two fixed English instructions and a limited table arrangement.
Other planned steps are passed to it as written and are expected to fail; it has not been
validated for real robots or arbitrary scenes.

vla-tokenizer/tokenizer.json and tokenizer_config.json are the SmolVLM2 tokenizer files from the
training checkpoint's processor (HuggingFaceTB/SmolVLM2-500M-Video-Instruct, Apache License 2.0),
used unchanged to tokenize instructions other than the two exported ones.

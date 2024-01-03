# Fine tune your own private Copilot
<a target="_blank" href="https://colab.research.google.com/github/seanses2/LLM_fine_tuning/blob/main/fine-tune-code-llama.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Introduction
The integration between GitHub and Colab has been annoyingly difficult. While it's possible to open a notebook from a GitHub link in Colab, unfortunately, none of the rest of the repository content is brought into the Colab runtime. This makes it cumbersome to make use of other materials saved in your repo, that includes your dataset preprocessing scripts, structured training code, and maybe even the dataset itself. People have compromised and resorted to alternative solutions to complete a fine tuning lifecycle:

1. First create some dataset and put it in GDrive or a Hugging Face dataset repo.
2. Put up some code in notebook and run it in Colab, loading models from a Hugging Face model repo.
3. Save the fine tuned model back into a Hugging Face model repo.
4. Evaluate the fine tuned model. And if it's not ideal, go back to step 1.

This breaks one project into three pieces stored in different places: a dataset repo, a source code (notebook) repo, and a model repo, and there's no good way to cross reference between their individual versions. For example, if one fine tuning lifecycle deteriorates, one has to manually search back into three parallel history, letting alone the difficulty to revert to a good base.

In this guide we demonstrate that one can
1. Version **all** three pieces together in one GitHub repo managed by [XetData](https://github.com/apps/XetData) GitHub app.
2. Clones **only** what you need in the training to Colab runtime using [Lazy clone](https://xethub.com/assets/docs/large-repos/lazy-clone) feature.


This fine tuning example uses a Lora approach on top of [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/), quantizing the base model to int 8, freezing its weights and only training an adapter. Please accept their License at https://ai.meta.com/resources/models-and-libraries/llama-downloads/. Much of the code is refactored from [[1]](https://github.com/tloen/alpaca-lora), [[2]](https://github.com/samlhuillier/code-llama-fine-tune-notebook/tree/main), [[3]](https://github.com/pacman100/DHS-LLM-Workshop/tree/main/personal_copilot).

## How to use this repository?
This repository already contains a drop of Code Llama in Hugging Face format. You can fork this repository and opens fine-tune-code-llama.ipynb in Colab. Follow the instructions in the notebook to fine tune your private Copilot and save it back to your repo!

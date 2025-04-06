# MIDI Multitrack Generation with Transformer-based GAN (WIP)

This is a personal research project where I explore generative approaches to symbolic music composition.  
The final architecture is a custom GAN built around a Transformer-based generator and a rhythm-aware discriminator, trained on piano roll representations of multi-track MIDI files.

---

## Why I'm doing this

I've been making music since 2013. When I began working with neural networks, I didn’t want to just *use* generative models — I wanted to **understand how they work from the inside**.

In this project, I:

- experimented with different architectures: RNNs (LSTM/GRU), Transformers, and Diffusion models  
- tried various tokenization strategies and training setups  
- eventually settled on a **GAN + Transformer** combo for its flexibility and multi-track control

---

## How I learned

This entire project was developed with GPT-4 as my assistant.  
But I didn’t just ask it to “write code” — I used it as a tutor.

I asked questions like:  
- what's the difference between tokens and tensors?  
- how does attention work, really?  
- what is autograd and why do we call backward()?  
- how is a VAE different from a vanilla autoencoder?

Based on these answers, I gradually learned how neural networks are built — and started crafting better prompts and better questions. This became my own process of AI education through GPT.

---

## What’s inside

- Multitrack symbolic generation based on piano rolls  
- Transformer-based generator with LSTM, segment, instrument and timing embeddings  
- Rhythm-aware discriminator  
- WGAN-GP loss  
- Top-k / Top-p sampling  
- TensorBoard logging and GradScaler optimization  
- Model scaled down to fit 8GB GPU memory

---

## Current status

- Project is **not yet fully functional**: training runs, architecture loads, but generation quality is low and dimensionality mismatches are still being resolved  
- Preprocessing and postprocessing improvements planned

---

## Why this matters

This project is part of my journey toward a deeper understanding of neural architectures.  
It’s not production-ready and not polished — but it’s mine.  
It reflects how I think, how I learn, how I iterate.

---

## Disclaimer

This repository is intended for learning and experimentation.  
It may break, crash, or generate weird music — but it’s a living project.

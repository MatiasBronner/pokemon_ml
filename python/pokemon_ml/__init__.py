"""Self-play training for Pokémon Champions doubles on the vgc_engine simulator.

    pokemon_ml.env      games stepped in bulk by the Rust engine, as NumPy arrays
    pokemon_ml.model    embeddings, a small transformer, policy and value heads
    pokemon_ml.ppo      rollouts, advantages and the PPO update
    pokemon_ml.train    the training loop (python -m pokemon_ml.train --help)
"""

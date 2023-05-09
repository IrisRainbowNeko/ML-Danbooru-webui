import launch

requirements=[
    'einops>=0.6.0',
    'huggingface_hub',
    'timm',
]

for req in requirements:
    launch.run_pip(f"install -U {req}", f"requirements for {req}")

import launch

requirements=[
    'einops>=0.6.0',
    'huggingface_hub',
    'timm',
]

launch.run_pip(f"uninstall einops", "re-install einops")

launch.run_pip(f"install -U {requirements[0]}", requirements[0])

for req in requirements[1:]:
    launch.run_pip(f"install {req}", req)

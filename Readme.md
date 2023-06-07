# ML-Danbooru: Anime image tags detector (webui extension)

## Introduction
An anime image taggger trained with cleaned [danbooru2021](https://gwern.net/danbooru2021).

[ML-Danbooru](https://github.com/7eu7d7/ML-Danbooru) uses the structure of multi-scale recognition, which has better detail recognition ability and more accurate than wd14.

## !!! 重要
使用这一插件需要修改WebUI的```requirements_versions.txt```文件，
将其中的```einops==0.4.1```替换为```einops>=0.6.0```!

## !!! Important
To use this extension, you need to modify the ```requirements_versions.txt``` file of WebUI.
Replace ```einops==0.4.1``` with ```einops>=0.6.0```!


![](./imgs/mld.webp)

## Model-Zoo
https://huggingface.co/7eu7d7/ML-Danbooru
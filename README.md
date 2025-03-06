# 🌔 Moondream2 Plugin


This plugin integrates Moondream2, a powerful vision-language model, into FiftyOne, enabling various visual AI capabilities like image captioning, visual question answering, object detection, and point localization.


### Plugin Overview

The plugin provides a seamless interface to Moondream2's capabilities within FiftyOne, offering:

* Multiple vision-language tasks:

  - Image captioning (short or detailed)

  - Visual question answering

  - Object detection

  - Point localization


* Hardware acceleration (CUDA/MPS) when available

* Dynamic version selection from HuggingFace

* Full integration with FiftyOne's Dataset and UI


## Installation

If you haven't already, install FiftyOne and required dependencies:

```bash
pip install -U fiftyone transformers torch pvips
```
On Ubuntu you may need to also install the pvips libraries

```bash
sudo apt install imagemagick libvips
```

Then, install the plugin:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/moondream2-plugin
```


## Usage in FiftyOne App

You can use Moondream2 directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset

2. Open the "Operators Browser" (click the icon or press `)
3. Search for "Run Moondream2"
4. Configure the parameters based on your chosen task:

### Available Tasks:

#### Image Captioning
- Choose between short or detailed captions
- Select model revision
- Specify output field name

#### Visual Question Answering
- Enter your question about the image
- Select model revision
- Specify output field name

#### Object Detection
- Specify the object type to detect
- Select model revision
- Specify output field name

#### Point Localization
- Specify the object to locate
- Select model revision
- Specify output field name

## Operator Usage via SDK

Once installed, you can use the operator programmatically:

```python

import fiftyone.operators as foo

moondream_operator = foo.get_operator("@harpreetsahota/moondream2/moondream")
```


# For image captioning

```python
moondream_operator(
    dataset,
    revision="2025-01-09",
    operation="caption",
    output_field="moondream_caption",
    length="normal"  # or "short"
)
```

# For visual question answering

```python
moondream_operator(
    dataset,
    revision="2025-01-09",
    operation="query",
    output_field="moondream_answer",
    query_text="What color is the car?"
)
```

# For object detection

```python
moondream_operator(
    dataset,
    revision="2025-01-09",
    operation="detect",
    output_field="moondream_detections",
    object_type="car"
)
```


# For point localization

```python
moondream_operator(
    dataset,
    revision="2025-01-09",
    operation="point",
    output_field="moondream_points",
    object_type="car"
)
```


If using delegated operation in an notebook, first run: `fiftyone delegated launch` and then use `await` with any of the operations.

```python
await moondream_operator(
    dataset,
    revision="2025-01-09",
    operation="caption",
    output_field="moondream_caption",
    length="normal",
    delegate=True
)
```

# Citation

Model weights are pulled from the [Moondream2 Hugging Face](https://huggingface.co/vikhyatk/moondream2) model card.

You can visit the original GitHub or the [Moondream website](https://moondream.ai/) for additional information.

```bibtex
@misc{moondream2024,
    author = {Korrapati, Vikhyat and others},
    title = {Moondream: A Tiny Vision Language Model},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/vikhyat/moondream},
    commit = {main}
}
```

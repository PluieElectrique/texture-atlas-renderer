# Texture Atlas Renderer

Render texture atlas animations exported from Adobe Animate.

## Install

You will need Python 3.6+ (tested on 3.9). Install dependencies with:

```
pip install -r requirements.txt
```

Note that if you look in `requirements.txt`, you'll see some installation options:

* If you install [PyAV](https://pyav.org/docs/stable/) from the binary wheel, then you'll be limited to the [features enabled in the built-in FFmpeg](https://pyav.org/docs/stable/overview/about.html#binary-wheels). If you want more features (e.g. codecs not included in the wheel), then [install PyAV from source](https://pyav.org/docs/stable/overview/about.html#bring-your-own-ffmpeg) with `--no-binary av`.
* You may get a small performance boost if you swap [Pillow](https://python-pillow.org/) for [Pillow-SIMD](https://github.com/uploadcare/pillow-simd). Note that [extra steps](https://github.com/uploadcare/pillow-simd#installation) are needed if you want to enable AVX2.
* You can skip [tqdm](https://tqdm.github.io/) if you don't want a progress bar.

## Usage

Export your texture atlas animation from Adobe Animate. You **must** enable "Optimize Animation.json", as this script will not work otherwise. Then, run:

```
python render.py animation-dir output.mp4
```

You can change the canvas size (the animation is always centered in the canvas), background color, or video codec with:

```
python render.py --width 1280 --height 720 --background-color "#fff" --codec x265 animation-dir output.mp4
```

There's also a `--resample` option, which can be `nearest`, `bilinear` (the default), or `bicubic`. I recommend leaving it at `bilinear`, as `nearest` is low quality and `bicubic` can add white pixels on the borders of sprites. Of course, your results may differ.

## Limitations

* In certain configurations of transformed and nested symbol instances, symbols may randomly change size. This is a bug in Animate's texture atlas export and can't be worked around.
* Animate can support fully transparent masks because it has access to the underlying shape information. But, when such a mask is exported as a sprite image, there is no shape information—it's just a fully transparent image. So, we can't apply these kinds of masks.

## Legal

This program is licensed under the MIT License. See the `LICENSE` file for more information.

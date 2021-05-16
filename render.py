if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render Adobe Animate texture atlas animations"
    )
    parser.add_argument(
        "in_dir",
        help="Directory containing Animation.json, spritemap1.json, and spritemap1.png",
    )
    parser.add_argument("out_file", help="Output video filename")
    parser.add_argument(
        "--width",
        default=500,
        type=int,
        help="Output video width (default: %(default)s)",
    )
    parser.add_argument(
        "--height",
        default=500,
        type=int,
        help="Output video height (default: %(default)s)",
    )
    parser.add_argument(
        "--background-color",
        default="#ccc",
        help="Animation background color (default: %(default)s)",
    )
    parser.add_argument(
        "--codec", default="h264", help="Video codec (default: %(default)s)"
    )
    parser.add_argument(
        "--resample",
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic"],
        help="Resampling filter (default: %(default)s)",
    )

    args = parser.parse_args()


from functools import lru_cache
import json
import math
import os
import queue
import threading
import warnings

import av
import numpy as np
from PIL import Image, ImageChops, ImageColor

try:
    from tqdm import trange
except ImportError:
    trange = range


# TransformMatrix and ColorEffect use the byte representations of their NumPy
# arrays for  __eq__ and __hash__. This is slightly faster than np.array_equal()
# and hash(tuple(array)). But, it's technically wrong because:
#   * 0 and -0 have different representations, so even though 0 == -0, they
#     won't be considered equal.
#   * NaNs should never equal NaNs, but identical NaNs will be equal here since
#     they have the same representation.
#
# However:
#   * -0 is unlikely to show up, and even if it does, at worst we'll only have
#     a cache miss.
#   * All NaNs completely break Pillow's affine transform, so it hardly matters
#     how we cache the result.


class TransformMatrix:
    """2D affine transformation matrix."""

    def __init__(self, m=None, a=1, b=0, c=0, d=0, e=1, f=0):
        if m is None:
            self.m = np.array(
                [
                    [a, b, c],
                    [d, e, f],
                    [0, 0, 1],
                ]
            )
        else:
            self.m = m

    @classmethod
    def parse(cls, m):
        """Create a TransformMatrix from an `M3D` list."""
        # The transformation matrix is in column-major order:
        # 0 4  8 12     a b 0 c
        # 1 5  9 13     d e 0 f
        # 2 6 10 14     0 0 1 0
        # 3 7 11 15     0 0 0 1
        return cls(a=m[0], b=m[4], c=m[12], d=m[1], e=m[5], f=m[13])

    def data(self):
        """Return a data tuple for Pillow's affine transformation."""
        # https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.AffineTransform
        #   "For each pixel (x, y) in the output image, the new value is taken
        #   from a position (a x + b y + c, d x + e y + f) in the input image,
        #   rounded to nearest pixel."
        #
        # Our matrix does the opposite (output = self.m @ input), so we invert
        # to get (self.m^-1 @ output = input).
        return np.linalg.inv(self.m).reshape(-1)[:6]

    def __eq__(self, other):
        return type(other) is TransformMatrix and self.m.tobytes() == other.m.tobytes()

    def __hash__(self):
        return hash(self.m.tobytes())

    def __matmul__(self, other):
        if type(other) is not TransformMatrix:
            raise TypeError(
                f"expected type TransformMatrix, but operand has type {type(other)}"
            )
        return TransformMatrix(m=self.m @ other.m)

    def __repr__(self):
        a, b, c, d, e, f = self.m.reshape(-1)[:6]
        return f"TransformMatrix(a={a}, b={b}, c={c}, d={d}, e={e}, f={f})"


class ColorEffect:
    """Color effects for RGBA images."""

    def __init__(self, effect=None):
        # None is a no-op effect
        self.effect = effect

    @classmethod
    def parse(cls, effect):
        """Create a ColorEffect from a `C` dictionary."""
        mode = effect["M"]
        # Advanced: multiply and offset each channel
        if mode == "AD":
            # Multipliers are in [-1, 1]
            multiplier = np.array(
                [
                    effect["RM"],
                    effect["GM"],
                    effect["BM"],
                    effect["AM"],
                ]
            )
            # Offsets are in [-255, 255]
            offset = np.array(
                [
                    effect["RO"],
                    effect["GO"],
                    effect["BO"],
                    effect["AO"],
                ]
            )
        # Alpha: multiply alpha by a constant
        elif mode == "CA":
            multiplier = np.array([1, 1, 1, effect["AM"]])
            offset = np.zeros(4)
        # Brightness: linearly interpolate towards black or white
        elif mode == "CBRT":
            brightness = effect["BRT"]
            if brightness < 0:
                # Linearly interpolate towards black
                multiplier = np.array(
                    [1 + brightness, 1 + brightness, 1 + brightness, 1]
                )
                offset = np.zeros(4)
            else:
                # Linearly interpolate towards white
                multiplier = np.array(
                    [1 - brightness, 1 - brightness, 1 - brightness, 1]
                )
                offset = brightness * np.array([255, 255, 255, 0])
        # Tint: linearly interpolate between the original color and a tint color
        elif mode == "T":
            tint_color = ImageColor.getrgb(effect["TC"])
            tint_multiplier = effect["TM"]
            # color * (1 - tint_multiplier) + tint_color * tint_multiplier
            multiplier = np.array(
                [1 - tint_multiplier, 1 - tint_multiplier, 1 - tint_multiplier, 1]
            )
            offset = tint_multiplier * np.array([*tint_color, 0])
        else:
            warnings.warn(f"Unsupported color effect: {effect}")
            return cls()

        return cls((multiplier, offset))

    def __call__(self, im):
        # An effect of None will return the input image.
        if self.effect is not None:
            mode = im.mode
            im = im.convert("RGBA")

            # We could use ImageMath to do everything in Pillow, but it's 4-5x
            # slower than NumPy. We clip to [0, 255] because that's what
            # Animate does. We need to cast to uint8 or the image will be
            # completely messed up.
            multiplier, offset = self.effect
            im = Image.fromarray(
                (np.array(im) * multiplier + offset).clip(0, 255).astype("uint8"),
                mode="RGBA",
            )

            im = im.convert(mode)

        return im

    def __eq__(self, other):
        if type(other) is not ColorEffect:
            return False
        elif self.effect is None or other.effect is None:
            return self.effect is other.effect
        else:
            multiplier_self, offset_self = self.effect
            multiplier_other, offset_other = other.effect
            return (
                multiplier_self.tobytes() == multiplier_other.tobytes()
                and offset_self.tobytes() == offset_other.tobytes()
            )

    def __hash__(self):
        if self.effect is None:
            return hash(None)
        else:
            multiplier, offset = self.effect
            return hash((multiplier.tobytes(), offset.tobytes()))

    def __matmul__(self, other):
        if type(other) is not ColorEffect:
            raise TypeError(
                f"expected type ColorEffect, but operand has type {type(other)}"
            )

        # ColorEffects are immutable, so it's fine to not always return a new instance.
        if self.effect is None:
            return other
        elif other.effect is None:
            return self
        else:
            # other is applied first, then self:
            #     m_self * (m_other * X + o_other) + o_self
            #   = (m_self * m_other) * X + (m_self * o_other + o_self)
            multiplier_self, offset_self = self.effect
            multiplier_other, offset_other = other.effect
            return ColorEffect(
                (
                    multiplier_self * multiplier_other,
                    multiplier_self * offset_other + offset_self,
                )
            )

    def __repr__(self):
        return f"ColorEffect({self.effect!r})"


class SpriteAtlas:
    """Extract and transform sprites."""

    def __init__(self, spritemap_json, img, canvas_size, resample):
        # Image.transform() only works with RGBa, not RGBA, so we pre-convert
        # for efficiency. (We'll have to convert back to RGBA for color
        # effects, but those are rare, so it's still faster.)
        if img.mode == "P":
            # Images with a palette can't be directly converted to RGBa
            img = img.convert("RGBA")
        img = img.convert("RGBa")

        self.img = img
        self.canvas_width, self.canvas_height = canvas_size
        self.resample = resample
        self.sprite_info = {}
        self.sprites = {}

        for sprite in spritemap_json["ATLAS"]["SPRITES"]:
            sprite = sprite["SPRITE"]
            x = sprite["x"]
            y = sprite["y"]
            w = sprite["w"]
            h = sprite["h"]
            self.sprite_info[sprite["name"]] = {
                "box": (x, y, x + w, y + h),
                "rotated": sprite["rotated"],
            }

    # On the animations I tested, 1024 gets almost all of the possible cache
    # hits, so there's little benefit to increasing maxsize further.
    @lru_cache(maxsize=1024)
    def get_sprite(self, name, m, color_effect):
        """Apply a transformation and color effect to a sprite.

        Image.transform() is very slow, and its slowness scales with the output
        size. So, before every .transform(), we add a translation to `m` so that
        the top-left corner of the transformed sprite's bbox is at the origin.
        This keeps the output image as small as possible.
        Thus, this method returns (sprite, (x, y)), where (x, y) is a
        translation offset to put the sprite back at the correct position.
        """
        if name not in self.sprites:
            sprite_info = self.sprite_info[name]
            sprite = self.img.crop(sprite_info["box"])
            if sprite_info["rotated"]:
                # Since "rotated" is a boolean, I'm guessing it's a fixed
                # rotation. I can't be bothered to figure out what the Unity
                # plugin is doing, but a 90 degree rotation seems to work.
                sprite = sprite.transpose(Image.ROTATE_90)
            self.sprites[name] = sprite
        else:
            sprite = self.sprites[name]

        # Find bounding box of transformed sprite by transforming the
        # coordinates of its corners.
        w, h = sprite.size
        corners = m.m @ np.array(
            [
                [0, w, 0, w],
                [0, 0, h, h],
                [1, 1, 1, 1],
            ]
        )

        # To be safe, we always round round the bbox to be bigger. (In
        # practice, the bbox is too big to begin with so it doesn't matter.)
        # min/max are a bit faster than np.min/np.max for tiny arrays.
        min_x = math.floor(min(corners[0]))
        max_x = math.ceil(max(corners[0]))
        min_y = math.floor(min(corners[1]))
        max_y = math.ceil(max(corners[1]))

        # Check if sprite is in bounds
        if (
            max_x < 0
            or self.canvas_width <= min_x
            or max_y < 0
            or self.canvas_height <= min_y
        ):
            warnings.warn(
                f"Sprite `{name}` is out of bounds, increase canvas size: "
                f"({min_x:.2f}, {min_y:.2f}) x ({max_x:.2f}, {max_y:.2f})"
            )
            return None, None

        # Clip against canvas viewport
        min_x = max(0, min_x)
        max_x = min(self.canvas_width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(self.canvas_height - 1, max_y)

        # We need to add 1 to get the correct width and height. For example, an
        # image with just one pixel at (0, 0) would have min_x = 0 and max_x =
        # 0. Not adding 1 would give a width of 0 - 0 = 0.
        transform_size = (max_x - min_x + 1, max_y - min_y + 1)

        # Bring top-left corner to (0, 0) to minimize output size.
        m = TransformMatrix(c=-min_x, f=-min_y) @ m

        # Apply color effect
        sprite = color_effect(sprite)

        sprite = sprite.transform(
            transform_size, Image.AFFINE, data=m.data(), resample=self.resample
        )

        # These sprites will be used by alpha_composite later, which only
        # accepts RGBA. So, we convert now to cache the RGBA image.
        return sprite.convert("RGBA"), (min_x, min_y)


class Symbols:
    """Render symbols."""

    def __init__(self, animation_json, sprite_atlas, canvas_size, background_color):
        self.background_color = ImageColor.getrgb(background_color)
        if len(self.background_color) == 4 and self.background_color[3] != 255:
            # I don't know exactly why there are artifacts, but oh well.
            # Transparent backgrounds don't seem useful or commonly supported
            # (GIF might be the main exception).
            warnings.warn(
                "Background color must be opaque or artifacts will appear. "
                f"Removing alpha of {self.background_color[-1]} from {background_color}."
            )
            self.background_color = self.background_color[:3]

        self.canvas_size = canvas_size
        self.sprite_atlas = sprite_atlas
        self.timelines = {}
        for symbol in animation_json["SD"]["S"]:
            name = symbol["SN"]
            layers = symbol["TL"]["L"]
            # I hope that symbol names are unique
            assert name not in self.timelines, "Symbol names should be unique"
            self.timelines[name] = layers

        # For convenience, we use None to store the main symbol. You could use
        # its actual name, but that's annoying to store and look up.
        assert None not in self.timelines
        self.timelines[None] = animation_json["AN"]["TL"]["L"]

        # Translate to canvas center. We use integer coordinates for the center
        # to ensure consistency between odd and even canvas sizes.
        self.center_in_canvas = TransformMatrix(
            c=canvas_size[0] // 2, f=canvas_size[1] // 2
        )

    def length(self, symbol_name):
        """The length of a symbol is the index of its final frame."""
        length = 0
        # We have to search through every layer because layers aren't
        # necessarily padded out with empty keyframes to the same length.
        for layer in self.timelines[symbol_name]:
            if layer["FR"]:
                last_frame = layer["FR"][-1]
                length = max(length, last_frame["I"] + last_frame["DU"])
        return length

    def render_symbol(self, name, frame_idx):
        """Render a symbol (on a certain frame) to an image."""
        canvas = Image.new("RGBA", self.canvas_size, color=self.background_color)
        self._render_symbol(
            canvas, name, frame_idx, self.center_in_canvas, ColorEffect()
        )
        return canvas

    # This method recursively traverses the symbol data, accumulating sprites
    # (onto `canvas`), transforms (into `m`), and color effects (into `color`)
    # as it goes. This is complex and the matrix multiplications can't be
    # cached, but it's the simplest way I could think of to support masking.
    def _render_symbol(self, canvas, name, frame_idx, m, color):
        # If this symbol has mask layers, we will need to deal with three
        # canvases: the canvas containing all non-masked sprites, the canvas
        # with the sprites to be masked, and the mask canvas.
        # They will be pushed to this stack in that order (except for the mask
        # canvas, which isn't pushed).
        canvas_stack = []

        # Layers are ordered from front to back. We reverse so that the symbol
        # will be ordered from back to front for rendering.
        for layer in reversed(self.timelines[name]):
            frames = layer["FR"]
            if not frames:
                continue

            # Find frame using binary search
            low = 0
            high = len(frames) - 1
            while low != high:
                mid = (low + high + 1) // 2
                if frame_idx < frames[mid]["I"]:
                    high = mid - 1
                else:
                    low = mid
            frame = frames[low]
            if not (frame["I"] <= frame_idx < frame["I"] + frame["DU"]):
                continue

            # The layer structure for masks is as follows:
            #
            #   ...
            #   Regular layer
            #   Mask layer ("LT": "Clp")
            #     Masked layer ("Clpb": mask layer name)
            #     Masked layer ("Clpb": mask layer name)
            #     ...
            #   Regular layer
            #   ...
            #
            # Remember that all of these layers are flattened into a list, and
            # that we go from bottom to top. (Also, masks cannot be nested
            # within the same symbol.)
            # So, we'll composite sprites as normal until we hit a masked
            # layer. Then, we'll push the current canvas and begin compositing
            # the sprites-to-be-masked on a new canvas. We'll do the same thing
            # when we hit the mask layer. When we're done compositing together
            # the mask, we'll apply it to the canvas-to-be-masked, and then
            # apply the result onto the base canvas.

            # If this is the bottommost masked layer or the mask layer itself,
            # then we should save the current canvas and swap in a new one.
            if ("Clpb" in layer and not canvas_stack) or layer.get("LT") == "Clp":
                canvas_stack.append(canvas)
                canvas = Image.new("RGBA", self.canvas_size, color=(0, 0, 0, 0))

            # Elements are ordered from back to front, so we don't reverse.
            for element in frame["E"]:
                # Symbol instance
                if "SI" in element:
                    element = element["SI"]
                    element_name = element["SN"]

                    if "C" in element:
                        element_color = color @ ColorEffect.parse(element["C"])
                    else:
                        element_color = color

                    self._render_symbol(
                        canvas,
                        element_name,
                        element["FF"],
                        m @ TransformMatrix.parse(element["M3D"]),
                        element_color,
                    )
                # Atlas sprite instance
                else:
                    element = element["ASI"]
                    element_name = element["N"]
                    sprite, dest = self.sprite_atlas.get_sprite(
                        element_name, m @ TransformMatrix.parse(element["M3D"]), color
                    )
                    if sprite is not None:
                        canvas.alpha_composite(sprite, dest=dest)

            # If this is a mask layer, we've finished compositing it, so it's
            # time to apply it.
            if layer.get("LT") == "Clp":
                mask_canvas = canvas
                masked_canvas = canvas_stack.pop()
                base_canvas = canvas_stack.pop()

                # Masks are usually small, so it's faster to first crop the
                # canvas to its visible region.
                mask_bbox = mask_canvas.getbbox()
                if mask_bbox is None:
                    # Animate will apply a mask even if it's completely
                    # transparent because it knows the shape of the mask. But,
                    # a transparent mask is exported as a truly transparent
                    # image, which means we can't apply the mask.
                    warnings.warn(
                        f"Mask `{layer.get('LN')}` in symbol `{name}` "
                        "is fully transparent and can't be applied."
                    )
                    base_canvas.alpha_composite(masked_canvas)
                else:
                    mask_canvas = mask_canvas.crop(mask_bbox)
                    masked_canvas = masked_canvas.crop(mask_bbox)
                    masked_alpha = masked_canvas.getchannel("A")

                    # A mask is supposed to ignore color/transparency and only
                    # care about whether a pixel is "filled" or not. But, all
                    # we have is an image, so we have to use alpha to determine
                    # the shape of the mask.
                    # We could just count non-zero alpha as opaque, but that
                    # may lead to harsh edges. So, to preserve antialiasing, we
                    # use the alpha as it is.
                    # The problem is that a color effect may have scaled alpha.
                    # So, we scale alpha's maximum back to 255.
                    # (It'd be better to skip color effects on mask layers, but
                    # that's complicated. At worst, this approach introduces a
                    # bit of quantization error by casting to uint8 twice.)
                    mask_alpha = np.array(mask_canvas.getchannel("A"))
                    mask_alpha = Image.fromarray(
                        (mask_alpha / np.max(mask_alpha) * 255)
                        .clip(0, 255)
                        .astype("uint8"),
                        "L",
                    )

                    # It's faster to convert mask_alpha to an image and use
                    # ImageChops than it is to convert masked_alpha to a NumPy
                    # array, multiply, and then convert back to an Image.
                    masked_canvas.putalpha(
                        ImageChops.multiply(masked_alpha, mask_alpha)
                    )
                    base_canvas.alpha_composite(masked_canvas, dest=mask_bbox[:2])

                # Restore the original canvas
                canvas = base_canvas


class Animation:
    """Render a texture atlas animation (or one particular symbol)."""

    def __init__(self, animation_dir, canvas_size, background_color, resample):
        """Create an Animation from a texture atlas export directory."""
        with open(os.path.join(animation_dir, "Animation.json")) as f:
            animation_json = json.load(f)

        with open(os.path.join(animation_dir, "spritemap1.json"), "rb") as f:
            # json can't handle BOM
            spritemap_json = json.loads(f.read().decode("utf-8-sig"))

        spritemap_img = Image.open(os.path.join(animation_dir, "spritemap1.png"))

        self.frame_rate = animation_json["MD"]["FRT"]
        self.sprite_atlas = SpriteAtlas(
            spritemap_json, spritemap_img, canvas_size, resample
        )
        self.symbols = Symbols(
            animation_json, self.sprite_atlas, canvas_size, background_color
        )

    def render(self, out_file, codec, width, height, symbol_name=None):
        """Render a symbol (by default, the whole animation) to a video."""
        if codec not in av.codecs_available:
            raise RuntimeError(f"Codec {codec} is not supported by this FFmpeg build.")

        symbol_length = self.symbols.length(symbol_name)

        # PyAV enables threading by default, but stream.encode() can still
        # block the main thread. So, we move all encoding to a separate thread.

        # Queue to send frames from the main thread to the video thread.
        q = queue.Queue()
        # We will abuse the frame queue to send exceptions from the video
        # thread to the main thread. This Event will be used to signal such an
        # exception.
        exception = threading.Event()

        def video_thread():
            try:
                with av.open(out_file, "w") as container:
                    stream = container.add_stream(codec, rate=self.frame_rate)
                    stream.width = width
                    stream.height = height

                    for _ in range(symbol_length):
                        frame = av.VideoFrame.from_image(q.get())
                        for packet in stream.encode(frame):
                            container.mux(packet)

                    for packet in stream.encode():
                        container.mux(packet)
            except Exception:
                import sys

                q.put(sys.exc_info())
                exception.set()

        t = threading.Thread(target=video_thread, daemon=True)
        t.start()

        for frame_idx in trange(symbol_length, unit="fr"):
            # If `exception` is set, then pull the data from the frame queue.
            if exception.is_set():
                while not q.empty():
                    e = q.get()
                    if isinstance(e, tuple):
                        raise e[1].with_traceback(e[2])

            q.put(self.symbols.render_symbol(None, frame_idx))

        t.join()


if __name__ == "__main__":
    RESAMPLE_FILTERS = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }

    animation = Animation(
        args.in_dir,
        (args.width, args.height),
        args.background_color,
        RESAMPLE_FILTERS[args.resample],
    )
    animation.render(args.out_file, args.codec, args.width, args.height)

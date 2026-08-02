# Watch accuracy — app icon

Source of truth: `svg/watch-accuracy-default.svg` (1024 viewBox, no rounding — iOS masks the squircle).

## Contents

- `AppIcon.appiconset/` — drop straight into Assets.xcassets. Single 1024 asset per appearance (light / dark / tinted), the iOS 17+ format.
- `svg/` — editable vector for all four cuts: default, dark, tinted, small.
- `png/` — rasterised sizes.
  - `watch-accuracy-default-*` — full detail: 1024, 180, 167, 152, 120.
  - `watch-accuracy-small-*` — simplified cut for small sizes (four indices, heavier hands, fatter band, no gloss): 120, 87, 80, 76, 60, 58, 40.
  - `watch-accuracy-dark-1024`, `watch-accuracy-tinted-1024` — appearance variants.

## Notes

- Dial is optically centred 8px high in the 1024 canvas; don't re-centre it geometrically.
- The amber band is the measured rate range straddling twelve; the bead is the true rate. Keep the white twelve index visible inside the band — it is the zero reference.
- Tinted cut is greyscale by design; iOS applies the user's tint to luminance.
- To re-export after editing an SVG, re-render at the sizes above with any vector tool; keep the same filenames.

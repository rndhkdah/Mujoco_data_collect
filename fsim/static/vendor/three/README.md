three.js r186 (npm `three@0.186.1`), MIT License (see LICENSE), vendored unmodified so the headset page
works without internet access (the Quest reaches the server through `adb reverse`).

| file | source |
|---|---|
| three.module.js, three.core.js | `build/` |
| addons/loaders/GLTFLoader.js | `examples/jsm/loaders/` |
| addons/utils/BufferGeometryUtils.js, SkeletonUtils.js | `examples/jsm/utils/` |

Update: download the same paths from https://cdn.jsdelivr.net/npm/three@<version>/ and bump the version here.

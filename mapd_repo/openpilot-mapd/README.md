# Openpilot mapd (chriscarlo fork)
Provides openpilot with map data. Fork of [pfeiferj/openpilot-mapd](https://github.com/pfeiferj/openpilot-mapd)
that adds sigmoid-baked per-node safe speeds (schema v1) for VTSC rally
co-pilot. Release binaries live on
[chriscarlo/mapd](https://github.com/chriscarlo/mapd/releases).

## Using
### Integrating With Openpilot
Each release has a pre-compiled static arm64 binary attached for use with
openpilot on a comma device. The consumer for this fork is
[chriscarlo/chauffeur](https://github.com/chriscarlo/chauffeur), specifically
`sunnypilot/mapd/mapd_installer.py` (downloads the binary) and
`sunnypilot/selfdrive/controls/lib/vision_turn_controller.py` (consumes
`MapCurvatures` + `MapPreCurveSpeeds` params).

### mapd inputs
Inputs are described in [docs/inputs.md](./docs/inputs.md).

### mapd outputs
Outputs are described in [docs/outputs.md](./docs/outputs.md).

## Build
This project uses [earthly](https://github.com/earthly/earthly/) for its build
system. To install earthly follow the instructions at the
[get earthly page](https://earthly.dev/get-earthly)

### Format Code
```bash
earthly +format
```

### Lint
```bash
earthly +lint
```

### Test
```bash
earthly +test
```

### Update Snapshot Tests
```bash
earthly +update-snapshots
```

### Build capnp Files
```bash
earthly +compile-capnp
```

### Build Release Binary
NOTE: This will be built for ARM64 to be used on a comma device and may not work
on your computer
```bash
earthly +build-release
```

### Build Binary
NOTE: This will be built for your current archetecture and may not work on a
comma device
```bash
earthly +build
```

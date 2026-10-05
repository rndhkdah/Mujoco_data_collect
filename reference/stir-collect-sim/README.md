# STIR Simulation Data Collection

Collect Kinova Gen3 demonstrations in MuJoCo and robosuite through Meta Quest WebXR teleoperation.

## Trained weights

The pretrained simulation policies used in the paper are available in the
[stir-libero-policies collection](https://huggingface.co/stir-replanning/stir-libero-policies).

## Quick start

```bash
cd ~/workspace/stir-collect-sim
~/workspace/libero_venv/bin/python sim/assets/build_gripper.py
bash stack.sh check
bash stack.sh start
# In the Quest browser, open http://localhost:8002.
```

## Common commands

| Command | Purpose |
| --- | --- |
| `bash stack.sh sim` | Start the collection server |
| `bash stack.sh status` | Show the server, port, asset, and output status |
| `bash stack.sh configs` | List available environment configurations |
| `bash stack.sh kill` | Stop the collection server |
| `bash stack.sh help` | List all stack commands |

The default dataset goes under `$SIM_DATA_DIR/data` and the server runs on port `8002`.

## Requirements

- Python 3.12 virtual environment with `robosuite==1.4.0`
- LIBERO checkout and MuJoCo/EGL-capable GPU
- Meta Quest 3S in developer mode and `adb`
- Generated Menagerie gripper asset via `sim/assets/build_gripper.py`

## Documentation

- [Setup](SETUP.md)
- [Architecture](ARCHITECTURE.md)

## License

Apache-2.0. See [LICENSE](LICENSE).

# DataInterpretor

A Python GUI for transferring measurement data and interfacing with a Mitsubishi PLC.
The application provides file selection, connection testing, and basic PLC input/output
monitoring.

## Features
- **Tkinter user interface** built in `AppUI` for selecting files, CSV output directory,
  entering network details, and viewing PLC status lamps
- **Configuration management** persists settings such as file paths, PLC addresses and ready-image
  parameters via `app_settings.json`
- **Ready image detection** with optional calibration requiring `opencv-python`, `numpy` and
  `Pillow`
- **PLC communication** via the optional `pymcprotocol` library

## Requirements
- Python 3 with Tkinter
- Optional: `pymcprotocol`, `opencv-python`, `numpy`, `Pillow`

Install optional packages as needed:
```bash
pip install pymcprotocol opencv-python numpy Pillow
```

## Usage
Run the application from the repository root:
```bash
python main.py
```
The program loads and saves configuration to `app_settings.json` in the same directory.

## Configuration
`app_settings.json` stores defaults such as the PLC IP/port, CSV directory, ready-image path,
and mappings for PLC inputs/outputs. Modify this file or use the settings dialogs within the app
to update values.

## License
This project does not currently specify a license.

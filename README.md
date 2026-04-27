# llama-server-manager

### A web-based GUI for managing and loading GGUF models with [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in `llama-server`.
![Screenshot of llama-server-manager WebUI.](/assets/images/llama-server-manager.png)

## Features

- **Model Management** — Add, edit, and delete GGUF model configurations via a clean web interface
- **Real-time Console** — Stream `llama-server` debug output live using WebSocket
- **GPU Configuration** — Customize GPU offloading layers, context size, generation tokens and custom flags per model
- **Server Control** — Load and unload models with a single click, with live status indicator
- **Persistent Config** — Model configurations are saved to `config.json`

## Prerequisites

- Python 3.8+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with `llama-server` binary

## Installation

1. Install the Python dependencies:

```bash
pip install flask flask-socketio
```

2. Make sure the `llama-server` binary is available. By default the app expects it at LLAMA_SERVER_PATH:

```
<user_home_path>/llama.cpp/build/bin/llama-server
```

Update these variables in `app.py` to match your system.
```
LLAMA_SERVER_HOST
LLAMA_SERVER_PORT
LLAMA_SERVER_PATH
```

## Running

```bash
python app.py
```

The web UI will be available at `http://localhost:5000`.

## Usage

1. Open the web UI in your browser
2. Click **"+"** to add a new model — provide the path to a `.gguf` file and configure GPU layers, context size, and max tokens
3. Select a model from the list on the left panel
4. Click **"Load Model"** to start `llama-server` with that configuration
5. The console panel streams real-time output from the server
6. Click **"Unload Model"** to stop the server

## API Endpoints

| Method | Endpoint       | Description               |
|--------|----------------|---------------------------|
| GET    | `/api/models`  | List all configured models |
| POST   | `/api/models`  | Add a new model           |
| PUT    | `/api/models/<name>` | Update a model config |
| DELETE | `/api/models/<name>` | Remove a model      |
| POST   | `/api/load`    | Load a model into server  |
| POST   | `/api/unload`  | Unload the running model  |
| GET    | `/api/status`  | Check if server is running |

## Configuration

Models are stored in `config.json` with the following structure:

```json
{
  "models": [
    {
      "model_name": "My Model",
      "path_to_gguf_file": "/path/to/model.gguf",
      "n_gpu_layers": -1,
      "n_ctx": 8192,
      "n_generate_tokens": 4096
    }
  ]
}
```

- `n_gpu_layers`: Number of layers to offload to GPU (`-1` = let llama-server automatically calculate & decide - usually recommended) 
- `n_ctx`: Context window size (`0` = full context, decided by model)
- `n_generate_tokens`: Maximum tokens to generate per request (`-1` = full context, decided by model)

## License

MIT License — see [LICENSE](LICENSE)

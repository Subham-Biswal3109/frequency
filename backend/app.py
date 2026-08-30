import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, send_from_directory
from flask_cors import CORS
from backend.config import WEB_DIR
from backend.api.routes import api_bp
from backend.api.routes_simulation import simulation_bp
from backend.api.routes_jamming import jamming_bp

app = Flask(__name__, static_folder=str(WEB_DIR))
CORS(app)

# Register API routes
app.register_blueprint(api_bp)
# Spectrum Simulation module — additive, does not affect any existing route.
app.register_blueprint(simulation_bp)
# RF Interference/Jamming Detector — a SEPARATE model/task from spectrum
# availability. Additive, does not affect any existing route.
app.register_blueprint(jamming_bp)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

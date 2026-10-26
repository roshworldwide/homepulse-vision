"""AWS Lambda entry point (container image, Function URL) via Mangum."""

import numpy as np
from mangum import Mangum

from .app import app
from .perception import pipeline

# Load the ONNX models during Lambda init (which runs with a full CPU burst) rather than on the first request.
pipeline.analyze(np.full((240, 320, 3), 128, np.uint8))

handler = Mangum(app, lifespan="off")

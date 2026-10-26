"""AWS Lambda entry point (container image, Function URL) via Mangum."""

from mangum import Mangum

from .app import app

handler = Mangum(app, lifespan="off")

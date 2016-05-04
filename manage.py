#!/usr/bin/env python
from app.settings import (
    PORT
)
from app.web import create_app

if __name__ == '__main__':
    # manager.run(threaded=True)
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, threaded=True)

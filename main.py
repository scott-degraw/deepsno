#!/usr/bin/env python3

import jsonargparse

from src.test import test
from src.train import train

if __name__ == "__main__":
    jsonargparse.CLI(components=[test, train])

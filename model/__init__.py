import sys
import os

ap = os.path.abspath(__file__)
for i in range(3):
    ap = os.path.dirname(ap)
if ap not in sys.path:
    sys.path.append(ap)
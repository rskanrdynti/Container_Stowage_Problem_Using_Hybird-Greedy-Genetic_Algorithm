import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta

# ==========================================
# 1. KONFIGURASI FISIK KAPAL
# ==========================================
VESSEL_GEOMETRY = [
    (1, 4, 86, 13.931, -3.66, 76.929, 0),(1, 4, 84, 11.34, -3.66, 76.929, 0),(1, 3, 84, 11.34, 3.66, 76.929, 0),
    (1, 1, 84, 11.34, 1.22, 76.929, 0),(1, 2, 84, 11.34, -1.22, 76.929, 0),(1, 2, 86, 13.931, -1.22, 76.929, 0),
    (1, 1, 86, 13.931, 1.22, 76.929, 0),(1, 3, 86, 13.931, 3.66, 76.929, 0),(1, 3, 88, 16.522, 3.66, 76.929, 0),
    (1, 1, 88, 16.522, 1.22, 76.929, 0),(1, 2, 88, 16.522, -1.22, 76.929, 0),(1, 4, 88, 16.522, -3.66, 76.929, 0),
    (1, 0, 88, 16.522, 0.0, 76.929, 0),(1, 0, 86, 13.931, 0.0, 76.929, 0),(1, 0, 84, 11.34, 0.0, 76.929, 0),
    (3, 6, 6, 6.709, -6.1, 72.459, 5),(3, 4, 6, 6.709, -3.66, 72.459, 5),(3, 2, 6, 6.709, -1.22, 72.459, 5),
    (3, 1, 6, 6.709, 1.22, 72.459, 5),(3, 3, 6, 6.709, 3.66, 72.459, 5),(3, 5, 6, 6.709, 6.1, 72.459, 5),
    (3, 3, 4, 4.118, 3.66, 72.459, 5),(3, 1, 4, 4.118, 1.22, 72.459, 5),(3, 2, 4, 4.118, -1.22, 72.459, 5),
    (3, 4, 4, 4.118, -3.66, 72.459, 5),(3, 2, 2, 1.527, -1.22, 72.459, 5),(3, 1, 2, 1.527, 1.22, 72.459, 5),
    (3, 6, 84, 11.34, -6.1, 72.459, 5),(3, 4, 84, 11.34, -3.66, 72.459, 5),(3, 2, 84, 11.34, -1.22, 72.459, 5),
    (3, 1, 84, 11.34, 1.22, 72.459, 5),(3, 3, 84, 11.34, 3.66, 72.459, 5),(3, 5, 84, 11.34, 6.1, 72.459, 5),
    (3, 6, 86, 13.931, -6.1, 72.459, 5),(3, 0, 84, 11.34, 0.0, 72.459, 5),(3, 0, 86, 13.931, 0.0, 72.459, 5),
    (3, 4, 86, 13.931, -3.66, 72.459, 5),(3, 2, 86, 13.931, -1.22, 72.459, 5),(3, 1, 86, 13.931, 1.22, 72.459, 5),
    (3, 3, 86, 13.931, 3.66, 72.459, 5),(3, 5, 86, 13.931, 6.1, 72.459, 5),(3, 5, 88, 16.522, 6.1, 72.459, 5),
    (3, 3, 88, 16.522, 3.66, 72.459, 5),(3, 1, 88, 16.522, 1.22, 72.459, 5),(3, 0, 88, 16.522, 0.0, 72.459, 5),
    (3, 2, 88, 16.522, -1.22, 72.459, 5),(3, 4, 88, 16.522, -3.66, 72.459, 5),(3, 6, 88, 16.522, -6.1, 72.459, 5),
    (5, 3, 2, 1.527, 3.66, 66.349, 3),(5, 1, 2, 1.527, 1.22, 66.349, 3),(5, 2, 2, 1.527, -1.22, 66.349, 3),
    (5, 4, 2, 1.527, -3.66, 66.349, 3),(5, 6, 4, 4.118, -6.1, 66.349, 3),(5, 4, 4, 4.118, -3.66, 66.349, 3),
    (5, 2, 4, 4.118, -1.22, 66.349, 3),(5, 1, 4, 4.118, 1.22, 66.349, 3),(5, 3, 4, 4.118, 3.66, 66.349, 3),
    (5, 5, 4, 4.118, 6.1, 66.349, 3),(5, 5, 6, 6.709, 6.1, 66.349, 3),(5, 3, 6, 6.709, 3.66, 66.349, 3),
    (5, 1, 6, 6.709, 1.22, 66.349, 3),(5, 2, 6, 6.709, -1.22, 66.349, 3),(5, 4, 6, 6.709, -3.66, 66.349, 3),
    (5, 6, 6, 6.709, -6.1, 66.349, 3),(5, 0, 84, 11.34, 0.0, 66.349, 3),(5, 0, 86, 13.931, 0.0, 66.349, 3),
    (5, 0, 88, 16.522, 0.0, 66.349, 3),(5, 2, 84, 11.34, -1.22, 66.349, 3),(5, 4, 84, 11.34, -3.66, 66.349, 3),
    (5, 6, 84, 11.34, -6.1, 66.349, 3),(5, 6, 86, 13.931, -6.1, 66.349, 3),(5, 6, 88, 16.522, -6.1, 66.349, 3),
    (5, 4, 88, 16.522, -3.66, 66.349, 3),(5, 4, 86, 13.931, -3.66, 66.349, 3),(5, 2, 86, 13.931, -1.22, 66.349, 3),
    (5, 2, 88, 16.522, -1.22, 66.349, 3),(5, 1, 88, 16.522, 1.22, 66.349, 3),(5, 1, 86, 13.931, 1.22, 66.349, 3),
    (5, 1, 84, 11.34, 1.22, 66.349, 3),(5, 3, 84, 11.34, 3.66, 66.349, 3),(5, 5, 84, 11.34, 6.1, 66.349, 3),
    (5, 5, 86, 13.931, 6.1, 66.349, 3),(5, 3, 86, 13.931, 3.66, 66.349, 3),(5, 3, 88, 16.522, 3.66, 66.349, 3),
    (5, 5, 88, 16.522, 6.1, 66.349, 3),(7, 6, 6, 6.709, -6.1, 59.699, 9),(7, 4, 6, 6.709, -3.66, 59.699, 9),
    (7, 2, 6, 6.709, -1.22, 59.699, 9),(7, 1, 6, 6.709, 1.22, 59.699, 9),(7, 1, 4, 4.118, 1.22, 59.699, 9),
    (7, 1, 2, 1.527, 1.22, 59.699, 9),(7, 3, 2, 1.527, 3.66, 59.699, 9),(7, 5, 2, 1.527, 6.1, 59.699, 9),
    (7, 5, 4, 4.118, 6.1, 59.699, 9),(7, 3, 4, 4.118, 3.66, 59.699, 9),(7, 3, 6, 6.709, 3.66, 59.699, 9),
    (7, 5, 6, 6.709, 6.1, 59.699, 9),(7, 2, 2, 1.527, -1.22, 59.699, 9),(7, 2, 4, 4.118, -1.22, 59.699, 9),
    (7, 4, 4, 4.118, -3.66, 59.699, 9),(7, 4, 2, 1.527, -3.66, 59.699, 9),(7, 6, 2, 1.527, -6.1, 59.699, 9),
    (7, 6, 4, 4.118, -6.1, 59.699, 9),(7, 4, 84, 11.34, -3.66, 59.699, 9),(7, 4, 86, 13.931, -3.66, 59.699, 9),
    (7, 4, 88, 16.522, -3.66, 59.699, 9),(7, 4, 90, 19.113, -3.66, 59.699, 9),(7, 2, 90, 19.113, -1.22, 59.699, 9),
    (7, 2, 88, 16.522, -1.22, 59.699, 9),(7, 2, 86, 13.931, -1.22, 59.699, 9),(7, 2, 84, 11.34, -1.22, 59.699, 9),
    (7, 0, 84, 11.34, 0.0, 59.699, 9),(7, 0, 86, 13.931, 0.0, 59.699, 9),(7, 0, 88, 16.522, 0.0, 59.699, 9),
    (7, 0, 90, 19.113, 0.0, 59.699, 9),(7, 1, 90, 19.113, 1.22, 59.699, 9),(7, 1, 88, 16.522, 1.22, 59.699, 9),
    (7, 1, 86, 13.931, 1.22, 59.699, 9),(7, 1, 84, 11.34, 1.22, 59.699, 9),(7, 3, 84, 11.34, 3.66, 59.699, 9),
    (7, 3, 86, 13.931, 3.66, 59.699, 9),(7, 3, 88, 16.522, 3.66, 59.699, 9),(7, 3, 90, 19.113, 3.66, 59.699, 9),
    (7, 5, 90, 19.113, 6.1, 59.699, 9),(7, 5, 88, 16.522, 6.1, 59.699, 9),(7, 5, 86, 13.931, 6.1, 59.699, 9),
    (7, 5, 84, 11.34, 6.1, 59.699, 9),(9, 6, 90, 19.113, -6.1, 53.589, 7),(9, 4, 90, 19.113, -3.66, 53.589, 7),
    (9, 2, 90, 19.113, -1.22, 53.589, 7),(9, 0, 90, 19.113, 0.0, 53.589, 7),(9, 1, 90, 19.113, 1.22, 53.589, 7),
    (9, 3, 90, 19.113, 3.66, 53.589, 7),(9, 5, 90, 19.113, 6.1, 53.589, 7),(9, 6, 88, 16.522, -6.1, 53.589, 7),
    (9, 4, 88, 16.522, -3.66, 53.589, 7),(9, 2, 88, 16.522, -1.22, 53.589, 7),(9, 0, 88, 16.522, 0.0, 53.589, 7),
    (9, 1, 88, 16.522, 1.22, 53.589, 7),(9, 3, 88, 16.522, 3.66, 53.589, 7),(9, 5, 88, 16.522, 6.1, 53.589, 7),
    (9, 6, 86, 13.931, -6.1, 53.589, 7),(9, 4, 86, 13.931, -3.66, 53.589, 7),(9, 2, 86, 13.931, -1.22, 53.589, 7),
    (9, 0, 86, 13.931, 0.0, 53.589, 7),(9, 1, 86, 13.931, 1.22, 53.589, 7),(9, 3, 86, 13.931, 3.66, 53.589, 7),
    (9, 5, 86, 13.931, 6.1, 53.589, 7),(9, 6, 84, 11.34, -6.1, 53.589, 7),(9, 4, 84, 11.34, -3.66, 53.589, 7),
    (9, 2, 84, 11.34, -1.22, 53.589, 7),(9, 0, 84, 11.34, 0.0, 53.589, 7),(9, 1, 84, 11.34, 1.22, 53.589, 7),
    (9, 3, 84, 11.34, 3.66, 53.589, 7),(9, 5, 84, 11.34, 6.1, 53.589, 7),(9, 6, 6, 6.709, -6.1, 53.589, 7),
    (9, 4, 6, 6.709, -3.66, 53.589, 7),(9, 2, 6, 6.709, -1.22, 53.589, 7),(9, 1, 6, 6.709, 1.22, 53.589, 7),
    (9, 3, 6, 6.709, 3.66, 53.589, 7),(9, 5, 6, 6.709, 6.1, 53.589, 7),(9, 6, 4, 4.118, -6.1, 53.589, 7),
    (9, 4, 4, 4.118, -3.66, 53.589, 7),(9, 2, 4, 4.118, -1.22, 53.589, 7),(9, 1, 4, 4.118, 1.22, 53.589, 7),
    (9, 3, 4, 4.118, 3.66, 53.589, 7),(9, 5, 4, 4.118, 6.1, 53.589, 7),(9, 6, 2, 1.527, -6.1, 53.589, 7),
    (9, 4, 2, 1.527, -3.66, 53.589, 7),(9, 2, 2, 1.527, -1.22, 53.589, 7),(9, 1, 2, 1.527, 1.22, 53.589, 7),
    (9, 3, 2, 1.527, 3.66, 53.589, 7),(9, 5, 2, 1.527, 6.1, 53.589, 7),(11, 6, 90, 19.113, -6.1, 46.469, 13),
    (11, 4, 90, 19.113, -3.66, 46.469, 13),(11, 2, 90, 19.113, -1.22, 46.469, 13),(11, 0, 90, 19.113, 0.0, 46.469, 13),
    (11, 1, 90, 19.113, 1.22, 46.469, 13),(11, 3, 90, 19.113, 3.66, 46.469, 13),(11, 5, 90, 19.113, 6.1, 46.469, 13),
    (11, 6, 88, 16.522, -6.1, 46.469, 13),(11, 4, 88, 16.522, -3.66, 46.469, 13),(11, 2, 88, 16.522, -1.22, 46.469, 13),
    (11, 0, 88, 16.522, 0.0, 46.469, 13),(11, 1, 88, 16.522, 1.22, 46.469, 13),(11, 3, 88, 16.522, 3.66, 46.469, 13),
    (11, 5, 88, 16.522, 6.1, 46.469, 13),(11, 6, 86, 13.931, -6.1, 46.469, 13),(11, 4, 86, 13.931, -3.66, 46.469, 13),
    (11, 2, 86, 13.931, -1.22, 46.469, 13),(11, 0, 86, 13.931, 0.0, 46.469, 13),(11, 1, 86, 13.931, 1.22, 46.469, 13),
    (11, 3, 86, 13.931, 3.66, 46.469, 13),(11, 5, 86, 13.931, 6.1, 46.469, 13),(11, 6, 84, 11.34, -6.1, 46.469, 13),
    (11, 4, 84, 11.34, -3.66, 46.469, 13),(11, 2, 84, 11.34, -1.22, 46.469, 13),(11, 0, 84, 11.34, 0.0, 46.469, 13),
    (11, 1, 84, 11.34, 1.22, 46.469, 13),(11, 3, 84, 11.34, 3.66, 46.469, 13),(11, 5, 84, 11.34, 6.1, 46.469, 13),
    (11, 6, 6, 6.709, -6.1, 46.469, 13),(11, 4, 6, 6.709, -3.66, 46.469, 13),(11, 2, 6, 6.709, -1.22, 46.469, 13),
    (11, 1, 6, 6.709, 1.22, 46.469, 13),(11, 3, 6, 6.709, 3.66, 46.469, 13),(11, 5, 6, 6.709, 6.1, 46.469, 13),
    (11, 6, 4, 4.118, -6.1, 46.469, 13),(11, 4, 4, 4.118, -3.66, 46.469, 13),(11, 2, 4, 4.118, -1.22, 46.469, 13),
    (11, 1, 4, 4.118, 1.22, 46.469, 13),(11, 3, 4, 4.118, 3.66, 46.469, 13),(11, 5, 4, 4.118, 6.1, 46.469, 13),
    (11, 6, 2, 1.527, -6.1, 46.469, 13),(11, 4, 2, 1.527, -3.66, 46.469, 13),(11, 2, 2, 1.527, -1.22, 46.469, 13),
    (11, 1, 2, 1.527, 1.22, 46.469, 13),(11, 3, 2, 1.527, 3.66, 46.469, 13),(11, 5, 2, 1.527, 6.1, 46.469, 13),
    (13, 6, 90, 19.113, -6.1, 40.349, 11),(13, 4, 90, 19.113, -3.66, 40.349, 11),(13, 2, 90, 19.113, -1.22, 40.349, 11),
    (13, 0, 90, 19.113, 0.0, 40.349, 11),(13, 1, 90, 19.113, 1.22, 40.349, 11),(13, 3, 90, 19.113, 3.66, 40.349, 11),
    (13, 5, 90, 19.113, 6.1, 40.349, 11),(13, 6, 88, 16.522, -6.1, 40.349, 11),(13, 4, 88, 16.522, -3.66, 40.349, 11),
    (13, 2, 88, 16.522, -1.22, 40.349, 11),(13, 0, 88, 16.522, 0.0, 40.349, 11),(13, 1, 88, 16.522, 1.22, 40.349, 11),
    (13, 3, 88, 16.522, 3.66, 40.349, 11),(13, 5, 88, 16.522, 6.1, 40.349, 11),(13, 6, 86, 13.931, -6.1, 40.349, 11),
    (13, 4, 86, 13.931, -3.66, 40.349, 11),(13, 2, 86, 13.931, -1.22, 40.349, 11),(13, 0, 86, 13.931, 0.0, 40.349, 11),
    (13, 1, 86, 13.931, 1.22, 40.349, 11),(13, 3, 86, 13.931, 3.66, 40.349, 11),(13, 5, 86, 13.931, 6.1, 40.349, 11),
    (13, 6, 84, 11.34, -6.1, 40.349, 11),(13, 4, 84, 11.34, -3.66, 40.349, 11),(13, 2, 84, 11.34, -1.22, 40.349, 11),
    (13, 0, 84, 11.34, 0.0, 40.349, 11),(13, 1, 84, 11.34, 1.22, 40.349, 11),(13, 3, 84, 11.34, 3.66, 40.349, 11),
    (13, 5, 84, 11.34, 6.1, 40.349, 11),(13, 6, 6, 6.709, -6.1, 40.349, 11),(13, 4, 6, 6.709, -3.66, 40.349, 11),
    (13, 2, 6, 6.709, -1.22, 40.349, 11),(13, 1, 6, 6.709, 1.22, 40.349, 11),(13, 3, 6, 6.709, 3.66, 40.349, 11),
    (13, 5, 6, 6.709, 6.1, 40.349, 11),(13, 6, 4, 4.118, -6.1, 40.349, 11),(13, 4, 4, 4.118, -3.66, 40.349, 11),
    (13, 2, 4, 4.118, -1.22, 40.349, 11),(13, 1, 4, 4.118, 1.22, 40.349, 11),(13, 3, 4, 4.118, 3.66, 40.349, 11),
    (13, 5, 4, 4.118, 6.1, 40.349, 11),(13, 6, 2, 1.527, -6.1, 40.349, 11),(13, 4, 2, 1.527, -3.66, 40.349, 11),
    (13, 2, 2, 1.527, -1.22, 40.349, 11),(13, 1, 2, 1.527, 1.22, 40.349, 11),(13, 3, 2, 1.527, 3.66, 40.349, 11),
    (13, 5, 2, 1.527, 6.1, 40.349, 11),(15, 4, 90, 19.113, -3.66, 33.719, 17),(15, 2, 90, 19.113, -1.22, 33.719, 17),
    (15, 0, 90, 19.113, 0.0, 33.719, 17),(15, 1, 90, 19.113, 1.22, 33.719, 17),(15, 3, 90, 19.113, 3.66, 33.719, 17),
    (15, 5, 90, 19.113, 6.1, 33.719, 17),(15, 4, 88, 16.522, -3.66, 33.719, 17),(15, 2, 88, 16.522, -1.22, 33.719, 17),
    (15, 0, 88, 16.522, 0.0, 33.719, 17),(15, 1, 88, 16.522, 1.22, 33.719, 17),(15, 3, 88, 16.522, 3.66, 33.719, 17),
    (15, 5, 88, 16.522, 6.1, 33.719, 17),(15, 4, 86, 13.931, -3.66, 33.719, 17),(15, 2, 86, 13.931, -1.22, 33.719, 17),
    (15, 0, 86, 13.931, 0.0, 33.719, 17),(15, 1, 86, 13.931, 1.22, 33.719, 17),(15, 3, 86, 13.931, 3.66, 33.719, 17),
    (15, 5, 86, 13.931, 6.1, 33.719, 17),(15, 4, 84, 11.34, -3.66, 33.719, 17),(15, 2, 84, 11.34, -1.22, 33.719, 17),
    (15, 0, 84, 11.34, 0.0, 33.719, 17),(15, 1, 84, 11.34, 1.22, 33.719, 17),(15, 3, 84, 11.34, 3.66, 33.719, 17),
    (15, 5, 84, 11.34, 6.1, 33.719, 17),(15, 6, 6, 6.709, -6.1, 33.719, 17),(15, 4, 6, 6.709, -3.66, 33.719, 17),
    (15, 2, 6, 6.709, -1.22, 33.719, 17),(15, 1, 6, 6.709, 1.22, 33.719, 17),(15, 3, 6, 6.709, 3.66, 33.719, 17),
    (15, 5, 6, 6.709, 6.1, 33.719, 17),(15, 6, 4, 4.118, -6.1, 33.719, 17),(15, 4, 4, 4.118, -3.66, 33.719, 17),
    (15, 2, 4, 4.118, -1.22, 33.719, 17),(15, 1, 4, 4.118, 1.22, 33.719, 17),(15, 3, 4, 4.118, 3.66, 33.719, 17),
    (15, 5, 4, 4.118, 6.1, 33.719, 17),(15, 6, 2, 1.527, -6.1, 33.719, 17),(15, 4, 2, 1.527, -3.66, 33.719, 17),
    (15, 2, 2, 1.527, -1.22, 33.719, 17),(15, 1, 2, 1.527, 1.22, 33.719, 17),(15, 3, 2, 1.527, 3.66, 33.719, 17),
    (15, 5, 2, 1.527, 6.1, 33.719, 17),(17, 6, 90, 19.113, -6.1, 27.589, 15),(17, 4, 90, 19.113, -3.66, 27.589, 15),
    (17, 2, 90, 19.113, -1.22, 27.589, 15),(17, 0, 90, 19.113, 0.0, 27.589, 15),(17, 1, 90, 19.113, 1.22, 27.589, 15),
    (17, 3, 90, 19.113, 3.66, 27.589, 15),(17, 5, 90, 19.113, 6.1, 27.589, 15),(17, 6, 88, 16.522, -6.1, 27.589, 15),
    (17, 4, 88, 16.522, -3.66, 27.589, 15),(17, 2, 88, 16.522, -1.22, 27.589, 15),(17, 0, 88, 16.522, 0.0, 27.589, 15),
    (17, 1, 88, 16.522, 1.22, 27.589, 15),(17, 3, 88, 16.522, 3.66, 27.589, 15),(17, 5, 88, 16.522, 6.1, 27.589, 15),
    (17, 6, 86, 13.931, -6.1, 27.589, 15),(17, 4, 86, 13.931, -3.66, 27.589, 15),(17, 2, 86, 13.931, -1.22, 27.589, 15),
    (17, 0, 86, 13.931, 0.0, 27.589, 15),(17, 1, 86, 13.931, 1.22, 27.589, 15),(17, 3, 86, 13.931, 3.66, 27.589, 15),
    (17, 5, 86, 13.931, 6.1, 27.589, 15),(17, 6, 84, 11.34, -6.1, 27.589, 15),(17, 4, 84, 11.34, -3.66, 27.589, 15),
    (17, 2, 84, 11.34, -1.22, 27.589, 15),(17, 0, 84, 11.34, 0.0, 27.589, 15),(17, 1, 84, 11.34, 1.22, 27.589, 15),
    (17, 3, 84, 11.34, 3.66, 27.589, 15),(17, 5, 84, 11.34, 6.1, 27.589, 15),(17, 6, 6, 6.709, -6.1, 27.589, 15),
    (17, 4, 6, 6.709, -3.66, 27.589, 15),(17, 2, 6, 6.709, -1.22, 27.589, 15),(17, 1, 6, 6.709, 1.22, 27.589, 15),
    (17, 3, 6, 6.709, 3.66, 27.589, 15),(17, 5, 6, 6.709, 6.1, 27.589, 15),(17, 6, 4, 4.118, -6.1, 27.589, 15),
    (17, 4, 4, 4.118, -3.66, 27.589, 15),(17, 2, 4, 4.118, -1.22, 27.589, 15),(17, 1, 4, 4.118, 1.22, 27.589, 15),
    (17, 3, 4, 4.118, 3.66, 27.589, 15),(17, 5, 4, 4.118, 6.1, 27.589, 15),(17, 6, 2, 1.527, -6.1, 27.589, 15),
    (17, 4, 2, 1.527, -3.66, 27.589, 15),(17, 2, 2, 1.527, -1.22, 27.589, 15),(17, 1, 2, 1.527, 1.22, 27.589, 15),
    (17, 3, 2, 1.527, 3.66, 27.589, 15),(17, 5, 2, 1.527, 6.1, 27.589, 15),(19, 6, 6, 6.709, -6.1, 20.969, 0),
    (19, 4, 6, 6.709, -3.66, 20.969, 0),(19, 2, 6, 6.709, -1.22, 20.969, 0),(19, 1, 6, 6.709, 1.22, 20.969, 0),
    (19, 3, 6, 6.709, 3.66, 20.969, 0),(19, 5, 6, 6.709, 6.1, 20.969, 0),(19, 6, 4, 4.118, -6.1, 20.969, 0),
    (19, 4, 4, 4.118, -3.66, 20.969, 0),(19, 2, 4, 4.118, -1.22, 20.969, 0),(19, 1, 4, 4.118, 1.22, 20.969, 0),
    (19, 3, 4, 4.118, 3.66, 20.969, 0),(19, 5, 4, 4.118, 6.1, 20.969, 0),(19, 4, 2, 1.527, -3.66, 20.969, 0),
    (19, 2, 2, 1.527, -1.22, 20.969, 0),(19, 1, 2, 1.527, 1.22, 20.969, 0),(19, 3, 2, 1.527, 3.66, 20.969, 0),
    (19, 4, 92, 21.704, -3.66, 20.969, 0),(19, 2, 92, 21.704, -1.22, 20.969, 0),(19, 0, 92, 21.704, 0.0, 20.969, 0),
    (19, 1, 92, 21.704, 1.22, 20.969, 0),(19, 3, 92, 21.704, 3.66, 20.969, 0),(19, 5, 92, 21.704, 6.1, 20.969, 0),
    (19, 4, 90, 19.113, -3.66, 20.969, 0),(19, 2, 90, 19.113, -1.22, 20.969, 0),(19, 0, 90, 19.113, 0.0, 20.969, 0),
    (19, 1, 90, 19.113, 1.22, 20.969, 0),(19, 3, 90, 19.113, 3.66, 20.969, 0),(19, 5, 90, 19.113, 6.1, 20.969, 0),
    (19, 6, 88, 16.522, -6.1, 20.969, 0),(19, 4, 88, 16.522, -3.66, 20.969, 0),(19, 2, 88, 16.522, -1.22, 20.969, 0),
    (19, 0, 88, 16.522, 0.0, 20.969, 0),(19, 1, 88, 16.522, 1.22, 20.969, 0),(19, 3, 88, 16.522, 3.66, 20.969, 0),
    (19, 5, 88, 16.522, 6.1, 20.969, 0),(19, 6, 86, 13.931, -6.1, 20.969, 0),(19, 4, 86, 13.931, -3.66, 20.969, 0),
    (19, 2, 86, 13.931, -1.22, 20.969, 0),(19, 0, 86, 13.931, 0.0, 20.969, 0),(19, 1, 86, 13.931, 1.22, 20.969, 0),
    (19, 3, 86, 13.931, 3.66, 20.969, 0),(19, 5, 86, 13.931, 6.1, 20.969, 0),(19, 6, 84, 11.34, -6.1, 20.969, 0),
    (19, 4, 84, 11.34, -3.66, 20.969, 0),(19, 2, 84, 11.34, -1.22, 20.969, 0),(19, 0, 84, 11.34, 0.0, 20.969, 0),
    (19, 1, 84, 11.34, 1.22, 20.969, 0),(19, 3, 84, 11.34, 3.66, 20.969, 0),(19, 5, 84, 11.34, 6.1, 20.969, 0),
    (21, 4, 92, 21.704, -3.66, 13.479, 0),(21, 2, 92, 21.704, -1.22, 13.479, 0),(21, 0, 92, 21.704, 0.0, 13.479, 0),
    (21, 1, 92, 21.704, 1.22, 13.479, 0),(21, 3, 92, 21.704, 3.66, 13.479, 0),(21, 4, 90, 19.113, -3.66, 13.479, 0),
    (21, 2, 90, 19.113, -1.22, 13.479, 0),(21, 0, 90, 19.113, 0.0, 13.479, 0),(21, 1, 90, 19.113, 1.22, 13.479, 0),
    (21, 3, 90, 19.113, 3.66, 13.479, 0),(21, 4, 88, 16.522, -3.66, 13.479, 0),(21, 2, 88, 16.522, -1.22, 13.479, 0),
    (21, 0, 88, 16.522, 0.0, 13.479, 0),(21, 1, 88, 16.522, 1.22, 13.479, 0),(21, 3, 88, 16.522, 3.66, 13.479, 0),
    (21, 4, 86, 13.931, -3.66, 13.479, 0),(21, 2, 86, 13.931, -1.22, 13.479, 0),(21, 0, 86, 13.931, 0.0, 13.479, 0),
    (21, 1, 86, 13.931, 1.22, 13.479, 0),(21, 3, 86, 13.931, 3.66, 13.479, 0),(21, 4, 84, 11.34, -3.66, 13.479, 0),
    (21, 2, 84, 11.34, -1.22, 13.479, 0),(21, 0, 84, 11.34, 0.0, 13.479, 0),(21, 1, 84, 11.34, 1.22, 13.479, 0),
    (21, 3, 84, 11.34, 3.66, 13.479, 0),(21, 4, 82, 9.3, -3.66, 13.479, 0),(21, 2, 82, 9.3, -1.22, 13.479, 0),
    (21, 0, 82, 9.3, 0.0, 13.479, 0),(21, 1, 82, 9.3, 1.22, 13.479, 0),(21, 3, 82, 9.3, 3.66, 13.479, 0),
    (1, 4, 90, 19.113, -3.66, 76.929, 0),(1, 2, 90, 19.113, -1.22, 76.929, 0),(1, 0, 90, 19.113, 0.0, 76.929, 0),
    (1, 1, 90, 19.113, 1.22, 76.929, 0),(1, 3, 90, 19.113, 3.66, 76.929, 0),
    (5, 6, 90, 19.113, -6.1, 66.349, 3),(5, 4, 90, 19.113, -3.66, 66.349, 3),(5, 2, 90, 19.113, -1.22, 66.349, 3),
    (5, 0, 90, 19.113, 0.0, 66.349, 3),(5, 1, 90, 19.113, 1.22, 66.349, 3),(5, 3, 90, 19.113, 3.66, 66.349, 3),
    (5, 5, 90, 19.113, 6.1, 66.349, 3),
]

# ==========================================
# 2. STRUKTUR DATA (identik dengan hybrid)
# ==========================================
class Container:
    __slots__ = ['id', 'weight', 'size']
    def __init__(self, id_name, weight, size=None):
        self.id = str(id_name)
        self.weight = float(weight)
        if size is not None:
            self.size = int(size)
        else:
            self.size = 40 if "CB" in self.id.upper() else 20

class Slot:
    __slots__ = ['bay', 'row', 'tier', 'vcg', 'tcg', 'lcg', 'pasangan_bay', 'container', 'below_idx', 'pair_idx']
    def __init__(self, b, r, t, v, tc, l, lb=0):
        self.bay = b; self.row = r; self.tier = t
        self.vcg = v; self.tcg = tc; self.lcg = l
        self.pasangan_bay = lb
        self.container = None
        self.below_idx = None
        self.pair_idx = None

class Ship:
    def __init__(self, lw, lvcg, ltcg, llcg, tw=0, tvcg=0, ttcg=0, tlcg=0):
        self.slots = []; self.filled_indices = []
        self.lightship_w = lw; self.lightship_vcg = lvcg
        self.lightship_tcg = ltcg; self.lightship_lcg = llcg
        self.tank_w = tw; self.tank_vcg = tvcg
        self.tank_tcg = ttcg; self.tank_lcg = tlcg
        self.slot_map = {}
        self._cache_valid = False
        self._cache_vcg = 0; self._cache_tcg = 0; self._cache_lcg = 0; self._cache_w = 0

    def invalidate_cache(self):
        self._cache_valid = False

    def calculate_stability(self):
        if self._cache_valid:
            return (self._cache_vcg, self._cache_tcg, self._cache_lcg, self._cache_w)
        lpp = 93.6
        midship = lpp / 2
        total_w = self.lightship_w + self.tank_w
        mv = (self.lightship_w * self.lightship_vcg) + (self.tank_w * self.tank_vcg)
        mt = (self.lightship_w * self.lightship_tcg) + (self.tank_w * self.tank_tcg)
        ml = (self.lightship_w * (midship - self.lightship_lcg)) + (self.tank_w * (midship - self.tank_lcg))
        for i in self.filled_indices:
            s = self.slots[i]
            c = s.container
            w_per_slot = c.weight if c.size == 20 else c.weight / 2
            total_w += w_per_slot
            mv += w_per_slot * s.vcg
            mt += w_per_slot * s.tcg
            ml += w_per_slot * (midship - s.lcg)
        self._cache_vcg = mv / total_w
        self._cache_tcg = mt / total_w
        self._cache_lcg = ml / total_w
        self._cache_w = total_w
        self._cache_valid = True
        return (self._cache_vcg, self._cache_tcg, self._cache_lcg, total_w)

    def build_neighbor_map(self):
        for i, s in enumerate(self.slots):
            bellows = [idx for idx, o in enumerate(self.slots) if o.bay == s.bay and o.row == s.row and o.tier < s.tier]
            if bellows:
                self.slots[i].below_idx = max(bellows, key=lambda idx: self.slots[idx].tier)
            if s.pasangan_bay != 0:
                pair_key = (s.pasangan_bay, s.row, s.tier)
                if pair_key in self.slot_map:
                    self.slots[i].pair_idx = self.slot_map[pair_key]

# ==========================================
# 3. LOGIKA PENEMPATAN (identik dengan hybrid)
# ==========================================
def can_place(ship, idx, container):
    slot = ship.slots[idx]
    if slot.container: return False
    if container.size == 40:
        if slot.pair_idx is None: return False
        p_idx = slot.pair_idx
        if ship.slots[p_idx].container: return False
        b1 = slot.below_idx
        b2 = ship.slots[p_idx].below_idx
        if (b1 is None) != (b2 is None): return False
        for s_idx in [idx, p_idx]:
            b_idx = ship.slots[s_idx].below_idx
            if b_idx is not None:
                bc = ship.slots[b_idx].container
                if not bc: return False
                if bc.size == 40:
                    if container.weight > bc.weight: return False
                else:
                    if (container.weight / 2) > (bc.weight + 2.0): return False
    else:
        b_idx = slot.below_idx
        if b_idx is not None:
            bc = ship.slots[b_idx].container
            if not bc or bc.size == 40 or container.weight > bc.weight: return False
    return True

# ==========================================
# 4. FUNGSI FITNESS (identik dengan hybrid)
# ==========================================
def _stab_penalty_teu(value, target, a_lin, a_quad, tier2_teu, tier3_teu, thresh2, thresh3, tol=0.0):
    d = max(0.0, abs(value - target) - tol)
    p = a_lin * d + a_quad * d * d
    if d > thresh2: p += tier2_teu
    if d > thresh3: p += tier3_teu
    return p

_BASE_OFFSET = 2000

def calculate_fitness(ship, n20, n40):
    v, t, l, _ = ship.calculate_stability()
    total_teu = n20 + (n40 * 2)
    reward = float(total_teu)
    v_penalty = _stab_penalty_teu(v, 9.0,  80.0, 60.0, 200.0, 400.0, 0.5, 1.0)
    l_penalty = _stab_penalty_teu(l, 2.0,  40.0, 20.0,  60.0, 150.0, 0.3, 1.0)
    t_penalty = _stab_penalty_teu(t, 0.0,  20.0, 15.0,  30.0,  80.0, 0.3, 1.0)
    return _BASE_OFFSET + reward - (v_penalty + l_penalty + t_penalty)

# ==========================================
# 5. DECODER GREEDY MURNI
# ==========================================
def decode_greedy_strategy(ship, conts_40, conts_20, sorted_keys):
    """
    Single-pass greedy decoder.

    Cara kerja:
    1. Reset kapal (hapus semua kontainer).
    2. Tempatkan semua 40ft (urutan sesuai conts_40) ke slot pertama valid.
    3. Tempatkan semua 20ft (urutan sesuai conts_20) ke slot pertama valid.
    4. Urutan slot ditentukan oleh sorted_keys (berbeda per strategi).

    Tidak ada evolusi, tidak ada populasi -- satu pass deterministik.
    Kualitas hasil bergantung sepenuhnya pada kualitas heuristik
    (urutan kontainer × urutan slot) yang digunakan.
    """
    # Reset kapal
    for s in ship.slots:
        s.container = None
    ship.filled_indices = []
    ship.invalidate_cache()

    n20, n40 = 0, 0

    # Tempatkan 40ft terlebih dahulu
    for cont in conts_40:
        for key in sorted_keys:
            idx = ship.slot_map[key]
            if can_place(ship, idx, cont):
                slot = ship.slots[idx]
                slot.container = cont
                ship.filled_indices.append(idx)
                # Pasang ke slot pasangan (40ft span 2 bay)
                p_idx = slot.pair_idx
                if p_idx is not None:
                    ship.slots[p_idx].container = cont
                    ship.filled_indices.append(p_idx)
                n40 += 1
                ship.invalidate_cache()
                break

    # Tempatkan 20ft
    for cont in conts_20:
        for key in sorted_keys:
            idx = ship.slot_map[key]
            if can_place(ship, idx, cont):
                ship.slots[idx].container = cont
                ship.filled_indices.append(idx)
                n20 += 1
                ship.invalidate_cache()
                break

    return ship, n20, n40

# ==========================================
# 6. DEFINISI STRATEGI GREEDY
# ==========================================
def build_strategies(ship, all_conts):
    """
    Membangun 15 kombinasi strategi greedy dari:
    - 3 cara sort kontainer  x  5 cara sort slot

    Setiap strategi menghasilkan urutan penempatan yang berbeda,
    memberi kesempatan menemukan solusi yang lebih baik.
    """
    TARGET_VCG = 9.0
    TARGET_AP  = 44.8   # LCG target dari buritan (= midship - 2.0)
    slots      = ship.slots
    slot_map   = ship.slot_map
    keys       = list(slot_map.keys())

    # ---- Urutan Kontainer (3 variasi) --------------------------------
    conts_40_heavy_first = sorted([c for c in all_conts if c.size == 40], key=lambda c: -c.weight)
    conts_20_heavy_first = sorted([c for c in all_conts if c.size == 20], key=lambda c: -c.weight)
    conts_40_light_first = sorted([c for c in all_conts if c.size == 40], key=lambda c:  c.weight)
    conts_20_light_first = sorted([c for c in all_conts if c.size == 20], key=lambda c:  c.weight)
    conts_40_mid_first   = sorted([c for c in all_conts if c.size == 40], key=lambda c: abs(c.weight - 20.0))
    conts_20_mid_first   = sorted([c for c in all_conts if c.size == 20], key=lambda c: abs(c.weight - 15.0))

    container_sorts = [
        ("CS-A: 40ft berat→ringan, 20ft berat→ringan",  conts_40_heavy_first, conts_20_heavy_first),
        ("CS-B: 40ft ringan→berat, 20ft ringan→berat",  conts_40_light_first, conts_20_light_first),
        ("CS-C: 40ft berat-sedang, 20ft berat-sedang",  conts_40_mid_first,   conts_20_mid_first),
    ]

    # ---- Urutan Slot (5 variasi) -------------------------------------
    def sk(sort_fn):
        return sorted(keys, key=sort_fn)

    slot_sorts = [
        # SS-1: Sama dengan Hybrid (baseline greedy)
        ("SS-1: Tier↑ → |LCG-44.8|↑ → |Row-3|↑  [Hybrid baseline]",
         sk(lambda k: (slots[slot_map[k]].tier,
                       abs(slots[slot_map[k]].lcg - TARGET_AP),
                       abs(k[1] - 3)))),

        # SS-2: Tier dulu, lalu balans transversal, lalu LCG
        ("SS-2: Tier↑ → |Row-3|↑ → |LCG-44.8|↑",
         sk(lambda k: (slots[slot_map[k]].tier,
                       abs(k[1] - 3),
                       abs(slots[slot_map[k]].lcg - TARGET_AP)))),

        # SS-3: LCG dulu (longitudinal), lalu balans, lalu tier
        ("SS-3: |LCG-44.8|↑ → |Row-3|↑ → Tier↑",
         sk(lambda k: (abs(slots[slot_map[k]].lcg - TARGET_AP),
                       abs(k[1] - 3),
                       slots[slot_map[k]].tier))),

        # SS-4: VCG proximity (slot rendah agar VCG rendah, mendekati 9m)
        ("SS-4: |VCG-9.0|↑ → |Row-3|↑ → |LCG-44.8|↑",
         sk(lambda k: (abs(slots[slot_map[k]].vcg - TARGET_VCG),
                       abs(k[1] - 3),
                       abs(slots[slot_map[k]].lcg - TARGET_AP)))),

        # SS-5: Skor stabilitas gabungan (VCG × 3 + TCG × 2 + LCG × 2)
        ("SS-5: Skor stabilitas gabungan (VCG*3 + TCG*2 + LCG*2)",
         sk(lambda k: (abs(slots[slot_map[k]].vcg - TARGET_VCG) * 3.0 +
                       abs(slots[slot_map[k]].tcg)              * 2.0 +
                       abs(slots[slot_map[k]].lcg - TARGET_AP)  * 2.0))),
    ]

    # ---- Kombinasi 3 × 5 = 15 strategi --------------------------------
    strategies = []
    for cs_name, c40, c20 in container_sorts:
        for ss_name, sorted_keys in slot_sorts:
            strategies.append({
                'name':        f"{cs_name}  |  {ss_name}",
                'conts_40':    c40,
                'conts_20':    c20,
                'sorted_keys': sorted_keys,
            })
    return strategies

# ==========================================
# 7. DISPLAY UTILITIES
# ==========================================
def clr():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def print_header():
    print(color("+==========================================================================+", "1;35"))
    print(color("|     SHIP STOWAGE PLAN OPTIMIZER  --  GREEDY MURNI (Pure Greedy) v1.0   |", "1;35"))
    print(color("|     Metode: Algoritma Greedy Saja (15 Strategi Heuristik)              |", "1;35"))
    print(color("+==========================================================================+", "1;35"))
    print()

def print_strategy_table(results):
    """Tampilkan tabel perbandingan semua strategi greedy."""
    print()
    print(color("+=====================================================================================+", "1;35"))
    print(color("|                   HASIL SEMUA STRATEGI GREEDY (15 Strategi)                        |", "1;35"))
    print(color("+====+========+===========+=========+============+=========+=========================+", "35"))
    print(color("| ID |  TEU   |  Fitness  |  VCG(m) |   TCG(m)   |  LCG(m) |  Status                 |", "35"))
    print(color("+====+========+===========+=========+============+=========+=========================+", "35"))
    best_idx = max(range(len(results)), key=lambda i: results[i]['fit'])
    for idx, res in enumerate(results):
        teu = res['n20'] + res['n40'] * 2
        v, t, l = res['vcg'], res['tcg'], res['lcg']
        v_ok = abs(v - 9.0) < 0.5; t_ok = abs(t) < 0.5; l_ok = abs(l - 2.0) < 1.0
        status = "[OK] BAIK" if (v_ok and t_ok and l_ok) else ("[~] CUKUP" if (v_ok or l_ok) else "[!] REVIEW")
        is_best = idx == best_idx
        row_color = "1;33" if is_best else "37"
        bmark = "[BEST]" if is_best else "      "
        print(color(f"| {idx+1:2d} | {teu:5d}  | {res['fit']:>9.0f} | {v:7.3f} | {t:10.4f} | {l:7.3f} | {bmark} {status:<15} |", row_color))
    print(color("+====+========+===========+=========+============+=========+=========================+", "35"))

def display_results_final(ship, n20, n40, exec_time, best_strategy_name):
    v, t, l, tot = ship.calculate_stability()
    cargo_w = tot - ship.lightship_w - ship.tank_w
    total_teu = n20 + (n40 * 2)
    def delta_indicator(value, target, threshold_ok=0.3, threshold_warn=1.0):
        d = abs(value - target)
        arrow = "^" if value > target else "v"
        if d < threshold_ok:    return color(f"{arrow} D{d:.3f}", "32")
        elif d < threshold_warn: return color(f"{arrow} D{d:.3f}", "33")
        else:                    return color(f"{arrow} D{d:.3f}", "31")
    print()
    print(color("+==========================================================================+", "1;32"))
    print(color("|         [BEST] HASIL TERBAIK -- GREEDY MURNI (Pure Greedy)              |", "1;32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|  [CARGO] RINGKASAN MUATAN  |  [STAB]  PARAMETER STABILITAS              |", "32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|", "32") + f"  20ft (CA)  : {n20:5d} unit    " + color("|", "32") + f"  VCG  = {v:7.3f} m   target 9.000 m  {delta_indicator(v, 9.0)}  " + color("|", "32"))
    print(color("|", "32") + f"  40ft (CB)  : {n40:5d} unit    " + color("|", "32") + f"  TCG  = {t:8.4f} m   target 0.000 m  {delta_indicator(t, 0.0)}  " + color("|", "32"))
    print(color("|", "32") + f"  Total TEU  : {total_teu:5d} TEU     " + color("|", "32") + f"  LCG  = {l:7.3f} m   target 2.000 m  {delta_indicator(l, 2.0)}  " + color("|", "32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|", "32") + f"  Berat Kargo: {cargo_w:7.2f} ton  " + color("|", "32") + f"  Total Displasemen      : {tot:10.2f} ton       " + color("|", "32"))
    print(color("|", "32") + f"  Lightship  : {ship.lightship_w:7.2f} ton  " + color("|", "32") + f"  Waktu Komputasi         : {exec_time:7.3f} detik          " + color("|", "32"))
    print(color("+============================+=============================================+", "32"))
    print(color(f"\n  [STRAT] Strategi Terbaik: {best_strategy_name[:70]}", "36"))

def display_bay_layout(ship):
    bays_available = sorted(list(set(s.bay for s in ship.slots)))
    for b in bays_available:
        sample_slot = next(s for s in ship.slots if s.bay == b)
        _rows_exist = set(s.row for s in ship.slots if s.bay == b)
        _ROW_ORDER = [5, 3, 1, 0, 2, 4, 6]
        rows_in_bay = [r for r in _ROW_ORDER if r in _rows_exist]
        tiers_in_bay = sorted(list(set(s.tier for s in ship.slots if s.bay == b)), reverse=True)
        filled_in_bay = sum(1 for s in ship.slots if s.bay == b and s.container is not None)
        total_in_bay = sum(1 for s in ship.slots if s.bay == b)
        print(color(f"\n+{'-'*80}+", "90"))
        bay_title = f" BAY {b:02d}  |  LCG: {sample_slot.lcg:+.3f}m  |  Terisi: {filled_in_bay}/{total_in_bay} slot"
        print(color("|", "90") + color(f" {bay_title:<78}", "1;35") + color("|", "90"))
        print(color(f"+{'-'*80}+", "90"))
        row_header = color("|", "90") + color(" TIER ", "90") + color("|", "90")
        for r in rows_in_bay:
            row_header += color(f" R{r:02d}  ", "90") + color("|", "90")
        print(row_header)
        print(color(f"+{'-'*80}+", "90"))
        for tier in tiers_in_bay:
            row_cells = color("|", "90")
            tier_label = color(f" T{tier:02d} ", "33")
            row_cells += tier_label + color("|", "90")
            for row in rows_in_bay:
                key = (b, row, tier)
                if key in ship.slot_map:
                    c = ship.slots[ship.slot_map[key]].container
                    if c:
                        label = color(f"{c.weight:4.1f}t", "1;35") if c.size == 40 else color(f"{c.weight:4.1f}t", "1;32")
                    else:
                        label = color("  .  ", "90")
                else:
                    label = color(" XXX ", "31")
                row_cells += f" {label} " + color("|", "90")
            print(row_cells)
        print(color(f"+{'-'*80}+", "90"))
    print()
    print(color("  Legenda: ", "90") +
          color("# 20ft (CA)", "1;32") + "  " +
          color("# 40ft (CB)", "1;35") + "  " +
          color(". Kosong", "90") + "  " +
          color("XXX Tidak Tersedia", "31"))

def export_to_excel(ship, filename):
    rows = []
    rows.append({'Slot ID': 'LIGHTSHIP', 'Container ID': '-', 'Weight (Ton)': ship.lightship_w,
                 'VCG': ship.lightship_vcg, 'TCG': ship.lightship_tcg, 'LCG': ship.lightship_lcg})
    rows.append({'Slot ID': 'TANK', 'Container ID': '-', 'Weight (Ton)': ship.tank_w,
                 'VCG': ship.tank_vcg, 'TCG': ship.tank_tcg, 'LCG': ship.tank_lcg})
    processed_slots = set()
    for s in sorted(ship.slots, key=lambda x: (x.tier, x.bay, x.row)):
        if s.container and (s.bay, s.row, s.tier) not in processed_slots:
            c = s.container
            weight_per_bay = c.weight if c.size == 20 else c.weight / 2
            rows.append({'Slot ID': f"B{s.bay:02d}R{s.row:02d}T{s.tier:02d}",
                         'Container ID': c.id, 'Weight (Ton)': weight_per_bay,
                         'Size': f"{c.size}ft", 'VCG': s.vcg, 'TCG': s.tcg, 'LCG': s.lcg})
            processed_slots.add((s.bay, s.row, s.tier))
            if c.size == 40 and s.pasangan_bay != 0:
                key_pasangan = (s.pasangan_bay, s.row, s.tier)
                if key_pasangan in ship.slot_map:
                    s2 = ship.slots[ship.slot_map[key_pasangan]]
                    rows.append({'Slot ID': f"B{s2.bay:02d}R{s2.row:02d}T{s2.tier:02d}",
                                 'Container ID': c.id, 'Weight (Ton)': weight_per_bay,
                                 'Size': f"{c.size}ft", 'VCG': s2.vcg, 'TCG': s2.tcg, 'LCG': s2.lcg})
                    processed_slots.add((s2.bay, s2.row, s2.tier))
    df = pd.DataFrame(rows)
    df['Momen Vertikal'] = df['Weight (Ton)'] * df['VCG']
    df['Momen Transversal'] = df['Weight (Ton)'] * df['TCG']
    df['Momen Longitudinal'] = df['Weight (Ton)'] * df['LCG']
    v, t, l, tot = ship.calculate_stability()
    summary_rows = [
        {'Slot ID': '', 'Container ID': ''},
        {'Slot ID': 'SUMMARY -- GREEDY MURNI', 'Container ID': ''},
        {'Slot ID': 'Total Weight', 'Container ID': f'{tot:.2f} ton'},
        {'Slot ID': 'VCG', 'Container ID': f'{v:.3f} m'},
        {'Slot ID': 'TCG', 'Container ID': f'{t:.4f} m'},
        {'Slot ID': 'LCG', 'Container ID': f'{l:.3f} m'},
    ]
    df_final = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    df_final.to_excel(filename, index=False)
    print(color(f"\n  [SAVED] File Excel tersimpan: {filename}", "32"))

def print_unified_summary(label, n20, n40, vcg, tcg, lcg):
    total_teu = n20 + n40 * 2
    reward = float(total_teu)
    v_p = _stab_penalty_teu(vcg, 9.0, 80.0, 60.0, 200.0, 400.0, 0.5, 1.0)
    l_p = _stab_penalty_teu(lcg, 2.0, 40.0, 20.0, 60.0, 150.0, 0.3, 1.0)
    t_p = _stab_penalty_teu(tcg, 0.0, 20.0, 15.0, 30.0, 80.0, 0.3, 1.0)
    unified = _BASE_OFFSET + reward - (v_p + l_p + t_p)
    def _indicator(val, target, ok_thresh=0.3, warn_thresh=1.0):
        d = abs(val - target)
        if d < ok_thresh:    return "[OK]  "
        elif d < warn_thresh: return "[WARN]"
        else:                 return "[FAIL]"
    print()
    print("+" + "=" * 62 + "+")
    print(f"|  {'UNIFIED FITNESS SUMMARY':^58}  |")
    print(f"|  Program : {label:<50}|")
    print("+" + "=" * 62 + "+")
    print(f"|  TEU dimuat  : {total_teu:>5}  ({n20}x20ft + {n40}x40ft){'':<19}|")
    print(f"|  VCG         : {vcg:>8.3f} m   target 9.000 m  D{abs(vcg-9.0):.3f}  {_indicator(vcg,9.0)}  |")
    print(f"|  TCG         : {tcg:>8.4f} m   target 0.000 m  D{abs(tcg):.4f}  {_indicator(tcg,0.0)}  |")
    print(f"|  LCG         : {lcg:>8.3f} m   target 2.000 m  D{abs(lcg-2.0):.3f}  {_indicator(lcg,2.0)}  |")
    print("+" + "=" * 62 + "+")
    print(f"|  Penalty VCG : {v_p:>8.2f} TEU                              |")
    print(f"|  Penalty LCG : {l_p:>8.2f} TEU                              |")
    print(f"|  Penalty TCG : {t_p:>8.2f} TEU                              |")
    print(f"|  Reward TEU  : {reward:>8.1f}                                 |")
    print(f"|  Offset      : {_BASE_OFFSET:>8.1f}                                 |")
    print("+" + "=" * 62 + "+")
    print(f"|  [=] UNIFIED FITNESS = {unified:>10.2f}                        |")
    print("+" + "=" * 62 + "+")

# ==========================================
# 8. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    clr()
    print_header()

    # 1. Bangun kapal
    ship = Ship(2943.63, 8.65, 0.0, 41.49)
    for b, r, t, v, tc, l, lb in VESSEL_GEOMETRY:
        s = Slot(b, r, t, v, tc, l, lb)
        ship.slots.append(s)
        ship.slot_map[(b, r, t)] = len(ship.slots) - 1
    ship.build_neighbor_map()

    # 2. Input Excel
    path_excel = "data_container.xlsx"
    print(color(f"  [READ] Membaca file: {os.path.basename(path_excel)}", "36"))
    df_in = pd.read_excel(path_excel)
    all_conts = [Container(r.iloc[1], r.iloc[2]) for _, r in df_in.iterrows() if not pd.isna(r.iloc[2])]
    all_conts.sort(key=lambda x: x.weight, reverse=True)
    c20 = sum(1 for c in all_conts if c.size == 20)
    c40 = sum(1 for c in all_conts if c.size == 40)

    # 3. Info konfigurasi
    print(color("+-------------------------------------------------------------------------+", "90"))
    print(color("|  KONFIGURASI OPTIMASI -- ALGORITMA GREEDY MURNI                         |", "90"))
    print(color("+-------------------------------------------------------------------------+", "90"))
    print(color(f"|  [CARGO] Total Kargo   : {len(all_conts):3d} kontainer  ({c20}x 20ft CA  +  {c40}x 40ft CB)|", "90"))
    print(color( "|  [ALGO]  Metode        : Greedy murni -- tanpa evolusi / populasi       |", "90"))
    print(color( "|  [STRAT] Strategi      : 15 kombinasi (3 sort kontainer x 5 sort slot)  |", "90"))
    print(color( "|  [TGT]   Target        : VCG=9.000m  TCG=0.000m  LCG=2.000m (jarak)    |", "90"))
    print(color( "|  [SPEED] Waktu ekspek  : < 5 detik (tanpa evolusi, deterministik)       |", "90"))
    print(color("+-------------------------------------------------------------------------+", "90"))
    print()
    input(color("    Tekan ENTER untuk memulai optimasi greedy...\n", "1;33"))

    start_time = time.time()

    # 4. Bangun semua strategi
    strategies = build_strategies(ship, all_conts)
    n_strat = len(strategies)
    print(color(f"  [RUN]  Menjalankan {n_strat} strategi greedy...\n", "36"))

    # 5. Jalankan semua strategi, tracking best
    all_results = []
    best_result = None

    for idx, strat in enumerate(strategies):
        ship_i, n20_i, n40_i = decode_greedy_strategy(
            ship, strat['conts_40'], strat['conts_20'], strat['sorted_keys']
        )
        fit_i = calculate_fitness(ship_i, n20_i, n40_i)
        v_i, t_i, l_i, _ = ship_i.calculate_stability()
        teu_i = n20_i + n40_i * 2

        result = {
            'strategy': strat['name'],
            'fit': fit_i, 'n20': n20_i, 'n40': n40_i,
            'vcg': v_i, 'tcg': t_i, 'lcg': l_i,
        }
        all_results.append(result)

        # Simpan salinan ship untuk strategi terbaik
        if best_result is None or fit_i > best_result['fit']:
            # Re-run untuk menyimpan state kapal terbaik
            best_result = result.copy()
            best_result['n20'] = n20_i
            best_result['n40'] = n40_i

        pct = (idx + 1) / n_strat * 100
        bar_fill = int(40 * (idx + 1) / n_strat)
        bar = '[' + '#' * bar_fill + '.' * (40 - bar_fill) + f'] {pct:5.1f}%'
        print(f"\r  {bar}  Strategi {idx+1:2d}/{n_strat}  TEU={teu_i}  Fit={fit_i:.0f}", end='', flush=True)

    print()  # Newline setelah progress bar

    exec_time = time.time() - start_time

    # 6. Re-run strategi terbaik untuk mendapatkan state kapal final
    best_strat_idx = max(range(len(all_results)), key=lambda i: all_results[i]['fit'])
    best_strat = strategies[best_strat_idx]
    best_ship, best_n20, best_n40 = decode_greedy_strategy(
        ship, best_strat['conts_40'], best_strat['conts_20'], best_strat['sorted_keys']
    )
    best_v, best_t, best_l, _ = best_ship.calculate_stability()

    # 7. Tampilkan hasil
    clr()
    print_header()
    print_strategy_table(all_results)
    display_results_final(best_ship, best_n20, best_n40, exec_time, best_strat['name'])

    print()
    print(color("=" * 82, "90"))
    print(color("  [MAP]  LAYOUT PENEMPATAN KONTAINER PER BAY", "1;35"))
    print(color("=" * 82, "90"))
    display_bay_layout(best_ship)

    # 8. Export Excel
    filename_exc = os.path.basename(path_excel)
    filename_no_ext = os.path.splitext(filename_exc)[0]
    output_file = f"E:/Documents/ITS/Math/Semester 5/Kerja Praktik/work/GREEDY_ONLY_{filename_no_ext}.xlsx"
    export_to_excel(best_ship, output_file)

    total_teu = best_n20 + best_n40 * 2
    best_fit  = all_results[best_strat_idx]['fit']
    print()
    print(color(f"  [BEST] Strategi Terbaik : #{best_strat_idx+1} (Fitness: {best_fit:.0f})", "1;33"))
    print(color(f"  [TEU]  TEU Dimuat       : {total_teu} ({best_n20}x20ft + {best_n40}x40ft)", "1;33"))
    print(color(f"  [STAB] VCG={best_v:.3f}m  TCG={best_t:.4f}m  LCG={best_l:.3f}m", "1;33"))
    print(color(f"  [TIME] Total Waktu      : {exec_time:.3f} detik ({n_strat} strategi)", "36"))
    print(color(f"  [FILE] Output           : {output_file}", "36"))
    print()
    print_unified_summary(
        "Program Greedy Murni -- 15 Strategi Heuristik",
        best_n20, best_n40,
        best_v, best_t, best_l
    )

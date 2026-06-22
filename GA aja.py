import random
import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count, Manager
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

def do_place(ship, idx, container):
    slot = ship.slots[idx]
    slot.container = container
    ship.filled_indices.append(idx)
    if container.size == 40:
        lb = slot.pasangan_bay
        pair_key = (lb, slot.row, slot.tier)
        if pair_key in ship.slot_map:
            p_idx = ship.slot_map[pair_key]
            ship.slots[p_idx].container = container
            ship.filled_indices.append(p_idx)
    ship.invalidate_cache()

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
# 5. DECODER GA MURNI -- BEST-SLOT SELECTION
# ==========================================
def decode_ga_only(ship, chromosome, stab_sorted_idxs, stab_sorted_40_idxs):
    """
    Decoder Algoritma Genetika Murni.

    PERBEDAAN UTAMA dari Hybrid:
    - Hybrid  : Untuk setiap kontainer, pilih slot PERTAMA yang valid
                berdasarkan urutan tetap (sorted_keys = greedy first-fit).
    - GA Murni: Untuk setiap kontainer, evaluasi kandidat slot, pilih slot
                TERBAIK berdasarkan dampak terhadap stabilitas kapal
                (best-fit berbasis stabilitas, bukan first-fit).

    Cara kerja:
    1. Slot diurutkan sebelumnya berdasarkan kedekatannya ke target stabilitas.
    2. Untuk setiap kontainer (urutan dari kromosom GA), scan kandidat slot.
    3. Hitung skor stabilitas inkremental untuk setiap slot kandidat yang valid:
           skor = -(|VCG_baru - 9.0| * 3 + |TCG_baru| * 2 + |LCG_baru - 2.0| * 2)
    4. Tempatkan kontainer di slot dengan skor TERTINGGI (bukan slot pertama valid).
    5. Update momen inkremental untuk evaluasi berikutnya.

    Parameter SCAN_LIMIT membatasi jumlah kandidat valid yang dievaluasi
    agar performa tetap reasonable, sambil tetap mencari yang terbaik.
    """
    SCAN_LIMIT = 40  # Periksa hingga 40 kandidat valid per kontainer

    for s in ship.slots:
        s.container = None
    ship.filled_indices = []
    ship.invalidate_cache()

    n20, n40 = 0, 0
    lpp = 93.6
    midship = lpp / 2

    # Momen inkremental -- dihitung sekali, diupdate per penempatan (O(1) per slot)
    mv = (ship.lightship_w * ship.lightship_vcg + ship.tank_w * ship.tank_vcg)
    mt = (ship.lightship_w * ship.lightship_tcg + ship.tank_w * ship.tank_tcg)
    ml = (ship.lightship_w * (midship - ship.lightship_lcg) +
          ship.tank_w * (midship - ship.tank_lcg))
    tw = ship.lightship_w + ship.tank_w

    for cont in chromosome:
        best_idx = None
        best_score = float('-inf')
        valid_checked = 0

        # Pilih kandidat yang sudah diurutkan berdasarkan tipe kontainer
        candidates = stab_sorted_40_idxs if cont.size == 40 else stab_sorted_idxs

        for idx in candidates:
            if not can_place(ship, idx, cont):
                continue

            valid_checked += 1
            slot = ship.slots[idx]
            w = cont.weight / 2 if cont.size == 40 else cont.weight

            # Hitung stabilitas proyeksi setelah penempatan (inkremental, tanpa full recalc)
            if cont.size == 40:
                if slot.pair_idx is None:
                    continue
                p = ship.slots[slot.pair_idx]
                nw = tw + 2 * w
                nv = (mv + w * slot.vcg + w * p.vcg) / nw
                nt = (mt + w * slot.tcg + w * p.tcg) / nw
                nl = (ml + w * (midship - slot.lcg) + w * (midship - p.lcg)) / nw
            else:
                nw = tw + w
                nv = (mv + w * slot.vcg) / nw
                nt = (mt + w * slot.tcg) / nw
                nl = (ml + w * (midship - slot.lcg)) / nw

            # Skor stabilitas: lebih tinggi = lebih baik
            # Penalti deviasi dari target: VCG=9.0m, TCG=0.0m, LCG_dist=2.0m
            score = -(abs(nv - 9.0) * 3.0 + abs(nt) * 2.0 + abs(nl - 2.0) * 2.0)

            if score > best_score:
                best_score = score
                best_idx = idx

            if valid_checked >= SCAN_LIMIT:
                break  # Cukup kandidat dievaluasi

        # Tempatkan di slot terbaik yang ditemukan
        if best_idx is not None:
            slot = ship.slots[best_idx]
            w = cont.weight / 2 if cont.size == 40 else cont.weight

            ship.slots[best_idx].container = cont
            ship.filled_indices.append(best_idx)
            mv += w * slot.vcg
            mt += w * slot.tcg
            ml += w * (midship - slot.lcg)
            tw += w

            if cont.size == 40:
                lb = slot.pasangan_bay
                pair_key = (lb, slot.row, slot.tier)
                if pair_key in ship.slot_map:
                    p_idx = ship.slot_map[pair_key]
                    p_slot = ship.slots[p_idx]
                    ship.slots[p_idx].container = cont
                    ship.filled_indices.append(p_idx)
                    mv += w * p_slot.vcg
                    mt += w * p_slot.tcg
                    ml += w * (midship - p_slot.lcg)
                    tw += w
                n40 += 1
            else:
                n20 += 1
            ship.invalidate_cache()

    return ship, n20, n40

# ==========================================
# 6. SINGLE GA RUN (GA MURNI)
# ==========================================
def run_single_ga(args):
    run_num, ship_tmplt_data, all_conts_data, pop_size, generations, base_seed, progress_queue = args

    random.seed(base_seed + run_num)
    np.random.seed(base_seed + run_num)

    # Bangun ulang ship untuk run ini
    ship_tmplt = Ship(
        ship_tmplt_data['lw'], ship_tmplt_data['lvcg'],
        ship_tmplt_data['ltcg'], ship_tmplt_data['llcg'],
        ship_tmplt_data['tw'], ship_tmplt_data['tvcg'],
        ship_tmplt_data['ttcg'], ship_tmplt_data['tlcg']
    )
    for slot_data in ship_tmplt_data['slots']:
        s = Slot(slot_data['bay'], slot_data['row'], slot_data['tier'],
                 slot_data['vcg'], slot_data['tcg'], slot_data['lcg'], slot_data['linkbay'])
        ship_tmplt.slots.append(s)
        ship_tmplt.slot_map[(s.bay, s.row, s.tier)] = len(ship_tmplt.slots) - 1

    all_conts = [Container(c['id'], c['weight'], c['size']) for c in all_conts_data]
    population = [random.sample(all_conts, len(all_conts)) for _ in range(pop_size)]

    ship_tmplt.build_neighbor_map()

    # -------------------------------------------------------
    # Pre-compute urutan slot berdasarkan kedekatan stabilitas
    # (berbeda dari hybrid yang menggunakan sorted_keys berbasis
    #  LCG-dari-AP dan row-centering secara terpisah)
    # -------------------------------------------------------
    TARGET_VCG, TARGET_AP = 9.0, 44.8
    all_idxs = list(range(len(ship_tmplt.slots)))

    # Slot diurutkan: semakin kecil skor deviasi stabilitas = lebih diutamakan
    stab_sorted_idxs = sorted(
        all_idxs,
        key=lambda i: (
            abs(ship_tmplt.slots[i].vcg - TARGET_VCG) * 3.0 +
            abs(ship_tmplt.slots[i].tcg) * 2.0 +
            abs(ship_tmplt.slots[i].lcg - TARGET_AP) * 2.0
        )
    )
    # Hanya slot yang mendukung 40ft (pair_idx tersedia)
    stab_sorted_40_idxs = [i for i in stab_sorted_idxs if ship_tmplt.slots[i].pair_idx is not None]

    best_overall = None
    no_improve_streak = 0
    EARLY_STOP_LIMIT = 5
    prev_best_fit = float('-inf')

    for gen in range(generations):
        scored = []
        for chrom in population:
            ship, n20, n40 = decode_ga_only(ship_tmplt, chrom, stab_sorted_idxs, stab_sorted_40_idxs)
            fit = calculate_fitness(ship, n20, n40)
            scored.append({'chrom': chrom, 'fit': fit, 'ship': ship, 'n20': n20, 'n40': n40})

        scored.sort(key=lambda x: x['fit'], reverse=True)

        if best_overall is None or scored[0]['fit'] > best_overall['fit']:
            best_overall = scored[0]

        if scored[0]['fit'] > prev_best_fit + 1.0:
            no_improve_streak = 0
            prev_best_fit = scored[0]['fit']
        else:
            no_improve_streak += 1

        if progress_queue is not None:
            cur_teu = best_overall['n20'] + best_overall['n40'] * 2
            progress_queue.put({
                'run': run_num, 'gen': gen + 1, 'total_gen': generations,
                'best_fit': best_overall['fit'], 'teu': cur_teu,
                'n20': best_overall['n20'], 'n40': best_overall['n40'],
                'stopped': False
            })

        if no_improve_streak >= EARLY_STOP_LIMIT:
            if progress_queue is not None:
                progress_queue.put({'run': run_num, 'gen': gen + 1, 'total_gen': generations,
                                    'best_fit': best_overall['fit'],
                                    'teu': best_overall['n20'] + best_overall['n40'] * 2,
                                    'n20': best_overall['n20'], 'n40': best_overall['n40'],
                                    'stopped': True, 'stop_gen': gen + 1})
            break

        # Evolusi: elitisme + crossover + mutasi adaptif
        elite_count = 8
        next_gen = [scored[i]['chrom'] for i in range(min(elite_count, len(scored)))]
        mut_rate = 0.15 + 0.15 * (no_improve_streak / EARLY_STOP_LIMIT)

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(scored[:max(12, pop_size // 2)], 2)
            size = len(p1['chrom'])
            cut = random.randint(1, size - 1)
            child = p1['chrom'][:cut] + [c for c in p2['chrom'] if c not in p1['chrom'][:cut]]
            if random.random() < mut_rate:
                i1, i2 = random.sample(range(size), 2)
                child[i1], child[i2] = child[i2], child[i1]
            next_gen.append(child)
        population = next_gen

    # Evaluasi final
    best_ship, f_n20, f_n40 = decode_ga_only(
        ship_tmplt, best_overall['chrom'], stab_sorted_idxs, stab_sorted_40_idxs
    )
    v, t, l, _ = best_ship.calculate_stability()

    return {
        'run': run_num, 'ship': best_ship, 'fit': best_overall['fit'],
        'n20': f_n20, 'n40': f_n40, 'vcg': v, 'tcg': t, 'lcg': l
    }

# ==========================================
# 7. DISPLAY UTILITIES (identik dengan hybrid)
# ==========================================
def clr():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def progress_bar(current, total, width=30, label="", fill='#', empty='.'):
    filled = int(width * current / total) if total > 0 else 0
    bar = fill * filled + empty * (width - filled)
    pct = 100 * current / total if total > 0 else 0
    return f"[{bar}] {pct:5.1f}% {label}"

def print_header():
    print(color("+==========================================================================+", "1;36"))
    print(color("|      SHIP STOWAGE PLAN OPTIMIZER  --  GA MURNI (Pure GA) v1.0          |", "1;36"))
    print(color("|      Metode: Algoritma Genetika Saja (Best-Slot Stability-Guided)       |", "1;36"))
    print(color("+==========================================================================+", "1;36"))
    print()

def print_config_summary(total_conts, c20, c40, n_runs, generations, pop_size, n_cpu):
    print(color("+-------------------------------------------------------------------------+", "90"))
    print(color("|  KONFIGURASI OPTIMASI -- ALGORITMA GENETIKA MURNI                       |", "90"))
    print(color("+----------------------+--------------------------------------------------+", "90"))
    print(color(f"|  [CARGO] Total Kargo     |", "90") + f"  {total_conts} kontainer  ({c20}x 20ft CA  +  {c40}x 40ft CB)       " + color("|", "90"))
    print(color(f"|  [GA]    GA Setup        |", "90") + f"  {n_runs} runs  x  {generations} generasi  x  pop {pop_size}               " + color("|", "90"))
    print(color(f"|  [CPU]   Paralel CPU     |", "90") + f"  {n_cpu} core(s) aktif                                   " + color("|", "90"))
    print(color(f"|  [TGT]   Target Stabilitas|", "90") + f"  VCG=9.000m  TCG=0.000m  LCG=2.000m               " + color("|", "90"))
    print(color(f"|  [ALGO]  Metode Decode   |", "90") + f"  GA Murni: Best-Slot (max 40 kandidat valid)        " + color("|", "90"))
    print(color(f"|  [ALGO]  Slot Sort       |", "90") + f"  |VCG-9|*3 + |TCG|*2 + |LCG-44.8|*2 (stabilitas)  " + color("|", "90"))
    print(color(f"|  [STOP]  Early Stopping  |", "90") + f"  5 generasi tanpa improvement                       " + color("|", "90"))
    print(color("+----------------------+--------------------------------------------------+", "90"))
    print()

def print_live_dashboard(run_states, total_runs, total_gen, global_best, elapsed):
    print(color("+-------------------------------------------------------------------------+", "90"))
    print(color("|  [LIVE] PROGRESS MONITOR -- GA MURNI                                   |", "90"))
    print(color("+------+----------------------------------+--------+----------------------+", "90"))
    print(color("| RUN  |  Generasi Progress               |  TEU   |  Fitness             |", "90"))
    print(color("+------+----------------------------------+--------+----------------------+", "90"))
    for run_id in sorted(run_states.keys()):
        st = run_states[run_id]
        gen_cur = st.get('gen', 0); gen_tot = st.get('total_gen', total_gen)
        teu = st.get('teu', 0); fit = st.get('best_fit', 0)
        stopped = st.get('stopped', False); stop_gen = st.get('stop_gen', gen_cur)
        if stopped:
            bar = progress_bar(stop_gen, gen_tot, width=20, fill='#', empty='-')
            status = color(f" [DONE] STOP@{stop_gen}", "32")
        elif gen_cur >= gen_tot:
            bar = progress_bar(gen_cur, gen_tot, width=20, fill='#', empty='-')
            status = color(" [DONE] SELESAI", "32")
        else:
            bar = progress_bar(gen_cur, gen_tot, width=20)
            status = color(f" [....] Gen {gen_cur:3d}/{gen_tot}", "33")
        fit_str = f"{fit:,.0f}" if fit != 0 else "-"
        print(f"| {color(f'R{run_id:02d}', '1')}  | {bar} | {teu:5d}  | {fit_str:>16}{status} |")
    print(color("+------+----------------------------------+--------+----------------------+", "90"))
    runs_done = sum(1 for s in run_states.values() if s.get('gen', 0) >= s.get('total_gen', total_gen) or s.get('stopped', False))
    if global_best:
        v, t, l = global_best.get('vcg', 0), global_best.get('tcg', 0), global_best.get('lcg', 0)
        teu_g = global_best.get('n20', 0) + global_best.get('n40', 0) * 2
        print(color(f"|  [BEST] GLOBAL BEST:  TEU={teu_g:3d}  VCG={v:.3f}  TCG={t:.4f}  LCG={l:.3f}         |", "1;33"))
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(color(f"|  [TIME] Waktu: {elapsed_str}  |  Runs Selesai: {runs_done}/{total_runs}                               |", "90"))
    print(color("+-------------------------------------------------------------------------+", "90"))

def print_comparison_table(all_results):
    print()
    print(color("+==========================================================================================+", "1;34"))
    print(color("|                    RINGKASAN SEMUA RUN -- GA MURNI                                       |", "1;34"))
    print(color("+=======+=======+===========+=========+============+=========+===========================+", "34"))
    print(color("|  RUN  |  TEU  |  Fitness  |  VCG(m) |   TCG(m)   |  LCG(m) |  Stabilitas               |", "34"))
    print(color("+=======+=======+===========+=========+============+=========+===========================+", "34"))
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]['fit'])
    for idx, res in enumerate(all_results):
        teu = res['n20'] + (res['n40'] * 2)
        v, t, l = res['vcg'], res['tcg'], res['lcg']
        v_ok = abs(v - 9.0) < 0.5; t_ok = abs(t) < 0.5; l_ok = abs(l - 2.0) < 1.0
        rating = "[OK]  BAIK" if (v_ok and t_ok and l_ok) else ("[~]   OK" if (v_ok or l_ok) else "[!!] PERLU REVIEW")
        is_best = idx == best_idx
        row_color = "1;33" if is_best else "37"
        best_marker = "[*]" if is_best else "  "
        print(color(f"| {best_marker}R{idx+1:02d} | {teu:5d} | {res['fit']:>9.0f} | {v:7.3f} | {t:10.4f} | {l:7.3f} |  {rating:<25} |", row_color))
    print(color("+=======+=======+===========+=========+============+=========+===========================+", "34"))

def display_results_final(ship, n20, n40, exec_time):
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
    print(color("|          [BEST] HASIL TERBAIK -- GA MURNI (Pure GA)                     |", "1;32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|  [CARGO] RINGKASAN MUATAN  |  [STAB]  PARAMETER STABILITAS              |", "32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|", "32") + f"  20ft (CA)  : {n20:5d} unit    " + color("|", "32") + f"  VCG  = {v:7.3f} m   target 9.000 m  {delta_indicator(v, 9.0)}  " + color("|", "32"))
    print(color("|", "32") + f"  40ft (CB)  : {n40:5d} unit    " + color("|", "32") + f"  TCG  = {t:8.4f} m   target 0.000 m  {delta_indicator(t, 0.0)}  " + color("|", "32"))
    print(color("|", "32") + f"  Total TEU  : {total_teu:5d} TEU     " + color("|", "32") + f"  LCG  = {l:7.3f} m   target 2.000 m  {delta_indicator(l, 2.0)}  " + color("|", "32"))
    print(color("+============================+=============================================+", "32"))
    print(color("|", "32") + f"  Berat Kargo: {cargo_w:7.2f} ton  " + color("|", "32") + f"  Total Displasemen      : {tot:10.2f} ton       " + color("|", "32"))
    print(color("|", "32") + f"  Lightship  : {ship.lightship_w:7.2f} ton  " + color("|", "32") + f"  Waktu Komputasi         : {exec_time:7.1f} detik          " + color("|", "32"))
    print(color("+============================+=============================================+", "32"))

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
        print(color("|", "90") + color(f" {bay_title:<78}", "1;36") + color("|", "90"))
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
        {'Slot ID': 'SUMMARY -- GA MURNI', 'Container ID': ''},
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
    ship_tmplt = Ship(2943.63, 8.65, 0.0, 41.49)
    for b, r, t, v, tc, l, lb in VESSEL_GEOMETRY:
        s = Slot(b, r, t, v, tc, l, lb)
        ship_tmplt.slots.append(s)
        ship_tmplt.slot_map[(b, r, t)] = len(ship_tmplt.slots) - 1

    # 2. Input Excel
    path_excel = "500 - beda 5 mix new.xlsx"
    print(color(f"  [READ] Membaca file: {os.path.basename(path_excel)}", "36"))
    df_in = pd.read_excel(path_excel)
    all_conts = [Container(r.iloc[1], r.iloc[2]) for _, r in df_in.iterrows() if not pd.isna(r.iloc[2])]
    all_conts.sort(key=lambda x: x.weight, reverse=True)
    c20 = sum(1 for c in all_conts if c.size == 20)
    c40 = sum(1 for c in all_conts if c.size == 40)

    # 3. Konfigurasi GA Murni
    # Catatan: POP_SIZE dan GENERATIONS sedikit dikurangi karena decode best-slot
    # lebih berat dari first-fit, namun kualitas per evaluasi lebih tinggi.
    JUMLAH_RUN  = 15
    POP_SIZE    = 30
    GENERATIONS = 40
    BASE_SEED   = 42
    N_CPU       = cpu_count()

    print_config_summary(len(all_conts), c20, c40, JUMLAH_RUN, GENERATIONS, POP_SIZE, N_CPU)
    input(color("    Tekan ENTER untuk memulai optimasi...\n", "1;33"))
    clr()
    print_header()

    # 4. Serialize data untuk multiprocessing
    ship_data = {
        'lw': ship_tmplt.lightship_w, 'lvcg': ship_tmplt.lightship_vcg,
        'ltcg': ship_tmplt.lightship_tcg, 'llcg': ship_tmplt.lightship_lcg,
        'tw': 0, 'tvcg': 0, 'ttcg': 0, 'tlcg': 0,
        'slots': [{'bay': s.bay, 'row': s.row, 'tier': s.tier, 'vcg': s.vcg,
                   'tcg': s.tcg, 'lcg': s.lcg, 'linkbay': s.pasangan_bay} for s in ship_tmplt.slots]
    }
    conts_data = [{'id': c.id, 'weight': c.weight, 'size': c.size} for c in all_conts]

    # 5. Manager progress real-time
    manager = Manager()
    progress_queue = manager.Queue()
    args_list = [(i+1, ship_data, conts_data, POP_SIZE, GENERATIONS, BASE_SEED, progress_queue)
                 for i in range(JUMLAH_RUN)]

    # 6. Jalankan paralel
    start_time = time.time()
    run_states = {}
    global_best_snapshot = None
    all_results = []

    with Pool(processes=N_CPU) as pool:
        async_result = pool.map_async(run_single_ga, args_list)
        last_refresh = 0
        REFRESH_RATE = 0.5
        while not async_result.ready():
            now = time.time()
            while not progress_queue.empty():
                try:
                    msg = progress_queue.get_nowait()
                    run_id = msg['run']
                    run_states[run_id] = msg
                except:
                    break
            if now - last_refresh >= REFRESH_RATE:
                elapsed = now - start_time
                clr()
                print_header()
                print_live_dashboard(run_states, JUMLAH_RUN, GENERATIONS, global_best_snapshot, elapsed)
                last_refresh = now
            time.sleep(0.1)
        all_results = async_result.get()

    exec_time = time.time() - start_time

    # 7. Pilih best & tampilkan
    best = max(all_results, key=lambda x: x['fit'])
    for res in all_results:
        rid = res['run']
        if rid not in run_states: run_states[rid] = {}
        run_states[rid]['teu'] = res['n20'] + res['n40'] * 2
        run_states[rid]['best_fit'] = res['fit']
        run_states[rid]['gen'] = GENERATIONS
        run_states[rid]['total_gen'] = GENERATIONS

    clr()
    print_header()
    print_comparison_table(all_results)
    display_results_final(best['ship'], best['n20'], best['n40'], exec_time)

    print()
    print(color("=" * 82, "90"))
    print(color("  [MAP]  LAYOUT PENEMPATAN KONTAINER PER BAY", "1;36"))
    print(color("=" * 82, "90"))
    display_bay_layout(best['ship'])

    # 8. Export Excel
    filename_exc = os.path.basename(path_excel)
    filename_no_ext = os.path.splitext(filename_exc)[0]
    output_file = f"E:/Documents/ITS/Math/Semester 5/Kerja Praktik/work/GA_ONLY_{filename_no_ext}.xlsx"
    export_to_excel(best['ship'], output_file)

    print()
    print(color(f"  [BEST] Run Terbaik : RUN {best['run']}  (Fitness: {best['fit']:.0f})", "1;33"))
    print(color(f"  [TIME] Total Waktu : {str(timedelta(seconds=int(exec_time)))}", "36"))
    print(color(f"  [FILE] Output      : {output_file}", "36"))
    print()
    print_unified_summary(
        "Program GA Murni -- Best-Slot Stability-Guided",
        best['n20'], best['n40'],
        best['vcg'], best['tcg'], best['lcg']
    )
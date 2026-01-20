from pymongo import MongoClient
import numpy as np

# -----------------------
# MongoDB Setup
# -----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["ir_db"]
docs_col = db["documents"]
index_col = db["inverted_index"]
from falkordb import FalkorDB
import redis.exceptions
try:
    db = FalkorDB(host='127.0.0.1', port=6379)
    kg = db.select_graph('agemis_kg')
    kg.delete()
    print("Graph deleted successfully to reset old dimension vectors.")
except Exception as e:
    print(f"Graph deletion warning (safe to ignore if not exists): {e}")
